"""Sokoban Speedrun"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import os
import queue
import random
import re
import statistics
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import reasoning_gym
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


IGNORE_INDEX = -100
# One node; trainers + vLLM generators must fit within it. Default is the benchmark's 8xH100;
# the env override supports smaller boxes (e.g. NODE_GPUS=4 with a smaller model). modal_app.py
# bakes the launch-time GPU allocation into the container env so this always matches the GPUs
# actually present.
NODE_GPUS = int(os.environ.get("NODE_GPUS", "8"))
# Interruption-based length control (ScaleRL A.10): appended to a still-thinking rollout that hit the
# generation cap, to force it to stop reasoning and emit a final answer instead of hard-truncating to 0.
# Ends with the OPEN <answer> tag: without it, models spend the whole forced-answer budget on a prose
# recap and never reach the tag (observed ~10% of late-run samples => guaranteed reward 0). The tag is
# output-format guidance (rules-legal), sits in the injected marker (loss-masked, no gradient), and the
# first-tag-wins extractor scores it exactly like a model-emitted tag.
INTERRUPTION_TEXT = " Okay, time is up. Let me stop thinking and formulate a final answer now.\n</think>\n\n<answer>"
TRUNC_EWMA_ALPHA = 0.3        # weight on the current step (0.7 on history): smooths to a *sustained* climb, not a transient
TRUNC_WARN_EWMA = 0.05        # ScaleRL's "keep <5%" line — log a warning at/above it
GNORM_EWMA_ALPHA = 0.3
GNORM_WARN_EWMA = 1.0         # ~2-4x healthy: log a gradient-climbing warning at/above it
ALLPASS_WARN_FRAC = 0.5      # all-pass zero-variance drops >= this share of fresh groups => train set
                             # being mastered (curriculum starvation / wasted generation); warn at/above it
WEIGHT_SYNC_TIMEOUT_S = 300.0  # bound the rank-0 weight broadcast so a dead vLLM child can't hang the trainer
ROLLOUT_GET_TIMEOUT_S = 300.0  # bound result-queue waits so a silent vLLM child crash can't stall the trainer
FIFO_HEAD_STALL_ABORT_S = 900.0  # abort if the next-in-file-order group never arrives while others flow: vLLM
                                 # schedules FCFS, so the head is always among the earliest-started requests and
                                 # this is far above any single group's worst-case latency (~6k tok under load)
# Trainer-group NCCL timeout. Workers sit in the Phase-C broadcast recv for ALL of rank 0's
# Phase A, which is legitimately unbounded (a reject storm needs >reject_limit consecutive groups,
# ~15-35 min of generation, before the diagnostic abort; the rank-0 setup workers wait on can also
# exceed 10 min on a cold node). torch's default 10-minute watchdog would kill healthy workers
# mid-recv with an opaque NCCL error before those paths resolve, so raise it well above any healthy
# stall; real failures are covered by the poison/init-status protocol and the explicit
# WEIGHT_SYNC/ROLLOUT_GET timeouts above.
TRAINER_PG_TIMEOUT_S = 7200.0
MSG_GENERATE = "generate"
MSG_ROLLOUT = "rollout"
MSG_INIT_WEIGHTS = "init_weights"
MSG_SYNC_WEIGHTS = "sync_weights"
MSG_SHUTDOWN = "shutdown"
MSG_ENGINE_READY = "engine_ready"
MSG_WEIGHTS_READY = "weights_ready"
MSG_ERROR = "error"
EVAL_PID_BASE = 1_000_000_000
# ============================ RUN RECIPE (single source of truth) ============================
# The benchmark sprint recipe lives here as a top-level constant. main() PREPENDS it to the CLI
# args, so a bare `python -m speedrun --run X --max-steps N` (or the Modal launcher, which only
# passes --run/--max-steps/--extra) reproduces it exactly. Any flag passed on the CLI appears AFTER
# the recipe and therefore OVERRIDES the recipe value (argparse last-wins). --run and --max-steps are
# deliberately NOT here (they vary per launch). Eval-only invocations also get the recipe prepended,
# so --eval-data below is also the eval set --eval-only scores against unless overridden.
# TRAINER COUNT: the canonical sprint runs NTRAINERS=3 (3 trainer ranks + 5 vLLM) — the trainer-bound
# sweet spot (measured at inflight 96; still trainer-bound at the FIFO ticket budget of 64, which only
# shrinks finished-inventory, not generation concurrency), ~30% faster per step than 2 trainers. That is a torchrun launch param,
# NOT a speedrun CLI arg, so it is pinned as the default in modal_app.py (NTRAINERS=3), not in this list.
RECIPE = [
    "--dtype", "float32",       # load trainer params/head fp32; train autocast runs body bf16, head/logits fp32
    "--train-data", "datasets/sokoban_train.jsonl",
    "--eval-data", "datasets/sokoban_eval.jsonl",
    "--examples-per-step", "16",
    "--num-samples", "8",
    "--temperature", "0.8",
    "--top-p", "0.95",
    "--max-new-tokens", "5632",
    "--max-model-len", "7168",
    "--device-batch-size", "2",
    "--logits-chunk-size", "2048",
    "--gradient-checkpointing",
    "--interruption",
    "--interrupt-answer-tokens", "512",
    "--learning-rate", "8e-7",
    "--init-lr-frac", "0.05",
    "--warmup-steps", "5",
    "--lr-schedule", "linear",
    "--min-lr-frac", "0.15",
    "--lr-decay-steps", "90",
    "--grad-clip", "1.0",
    "--cispo-eps", "4.0",
    "--loss-normalization", "sequence",  # sample-level (GRPO).
    "--max-staleness", "4",          # less off-policy than ScaleRL's 8; a proven stability lever
    "--inflight-requests", "64",     # outstanding-group tickets, reissued at CONSUME time: pins total unconsumed
                                     # inventory, so worst-case FIFO staleness is 64/(16 groups/step) = max-staleness.
                                     # Generation is unaffected: the node is trainer-bound (~20 groups actually
                                     # generating; the rest of the old 96 sat finished in result_q) and the engine
                                     # still sees up to 64*8 = 512 seqs = --vllm-max-num-seqs at cold start.
    "--vllm-gpu-mem-util", "0.9",
    "--vllm-stream-interval", "8",   # buffer token streaming to cut host/IPC overhead; no sampling/budget change
    "--vllm-max-num-batched-tokens", "8192",
    "--vllm-max-num-seqs", "512",
    "--eval-vllm-max-num-batched-tokens", "16384",
    "--eval-vllm-max-num-seqs", "1024",
    "--eval-concurrency", "1024",
    "--eval-top-p", "0.95",
    "--save-every", "0",
    "--save-final",
    "--wandb-rollout-samples", "20",  # rollouts/step logged to the wandb "rollouts_live" + final "rollouts" tables
]
# ============================================================================================
# ============================ SOKOBAN DOMAIN: prompt, answer extraction, scoring, dataset ============================
_BOARD_PLACEHOLDER = "{board}"
SOKOBAN_RG_PROMPT_TEMPLATE = """You are solving Sokoban under final-state scoring.

The board is shown with 0-indexed row numbers down the left side and column numbers across the top, so every cell has an address (row, col). Rows increase downward and columns increase rightward.

Symbols:
* = player on floor
% = player on goal
@ = box on floor
$ = box on goal
X = empty goal
+ = wall
- = empty floor

Basic rules:
- Moves are U, D, L, R.
- U = up (row - 1), D = down (row + 1), L = left (col - 1), R = right (col + 1).
- Each move attempts to move the player one cell in that direction.
- The player may move onto empty floor (-) or an empty goal (X), but not through a wall (+).
- If the target cell contains a box (@ or $), the player tries to push that box one cell in the same direction.
- A box can be pushed only if the cell beyond it is empty floor (-) or an empty goal (X).
- A box cannot be pushed into a wall, another box, or off the board.
- Boxes can only be pushed, never pulled.
- Invalid or blocked moves are ignored.

State rules:
- Boxes are @ and $. Goals are X, %, and $.
- Moving off % leaves X behind.
- Moving off * leaves - behind.
- Pushing a $ off its goal leaves X behind.
- Pushing an @ onto a goal makes it $.
- Pushing a $ onto another goal keeps it $.
- Pushing any box onto floor makes it @.
- A $ is not fixed; it is a movable box that is currently on a goal.

The puzzle is solved only if, after executing the entire move string, there are no @ boxes left. Do not append extra legal moves after solving, because they can make the final state unsolved.

Think through the board, simulate your move string, then give exactly one final line that wraps your move string in answer tags, like `<answer>...</answer>`. Put only your U/D/L/R moves between the tags.

Here is your puzzle:
{board}
"""


def format_board_with_axes(board: str) -> str:
    """Render a raw space-separated Sokoban board with 0-indexed row/column labels."""
    rows = [line.split() for line in board.splitlines() if line.strip()]
    n_cols = max((len(row) for row in rows), default=0)
    gutter = max(len(str(len(rows) - 1)), 1) if rows else 1

    def cell(text: str) -> str:
        return f"{text:>2}"

    header = " " * (gutter + 1) + " ".join(cell(str(c)) for c in range(n_cols))
    body = [
        f"{row_idx:>{gutter}} " + " ".join(cell(c) for c in row)
        for row_idx, row in enumerate(rows)
    ]
    return "\n".join([header, *body])


def render_sokoban_rg_prompt(board: str) -> str:
    return SOKOBAN_RG_PROMPT_TEMPLATE.replace(_BOARD_PLACEHOLDER, format_board_with_axes(board))


def extract_sokoban_board(question: str) -> str:
    """Accept board-only dataset records, with backward compatibility for old RG questions."""
    return question.split("Here is your puzzle:", 1)[-1].strip()


def build_sokoban_prompt(question: str) -> str:
    """Render the canonical Sokoban prompt."""
    board = extract_sokoban_board(question)
    return render_sokoban_rg_prompt(board)


SOKOBAN_MARKER_RE = re.compile(r"####")
SOKOBAN_MOVE_STRING_RE = re.compile(r"[UDLR]+")
SOKOBAN_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
SOKOBAN_ACTION_TOKEN_RE = re.compile(r"\b(?:up|down|left|right|[UDLR])\b", re.IGNORECASE)
SOKOBAN_ACTION_SEPARATOR_RE = re.compile(
    r"\b(?:up|down|left|right|[UDLR])\b|[\s,;|:\-.]+",
    re.IGNORECASE,
)
SOKOBAN_ACTION_MAP = {
    "U": "U",
    "UP": "U",
    "D": "D",
    "DOWN": "D",
    "L": "L",
    "LEFT": "L",
    "R": "R",
    "RIGHT": "R",
}


def normalize_sokoban_moves(candidate: str) -> str | None:
    compact = "".join(candidate.split()).upper()
    if compact and SOKOBAN_MOVE_STRING_RE.fullmatch(compact):
        return compact

    tokens = SOKOBAN_ACTION_TOKEN_RE.findall(candidate)
    if not tokens:
        return None
    leftover = SOKOBAN_ACTION_SEPARATOR_RE.sub("", candidate)
    if leftover:
        return None
    return "".join(SOKOBAN_ACTION_MAP[token.upper()] for token in tokens)


def find_sokoban_answer_end_and_moves(text: str) -> tuple[int | None, str | None]:
    """Locate the answer that scoring uses, as (answer_end, moves).

    SINGLE SOURCE OF TRUTH for answer extraction. Precedence: the FIRST <answer> tag wins;
    only when no tag exists anywhere does the first '####' marker's first non-blank
    line count. `moves` is what gets scored; `answer_end` is the
    char offset just past the scored answer, for --trim-after-answer. answer_end is non-None
    only when the answer parsed (an unparseable answer leaves the sequence untrimmed)."""
    tag = SOKOBAN_ANSWER_TAG_RE.search(text)
    if tag is not None:
        moves = normalize_sokoban_moves(tag.group(1))
        return (tag.end() if moves is not None else None), moves
    marker = SOKOBAN_MARKER_RE.search(text)
    if marker is None:
        return None, None
    pos = marker.end()
    for line in text[pos:].splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if not content.strip():
            pos += len(line)
            continue
        moves = normalize_sokoban_moves(content)
        return (pos + len(content) if moves is not None else None), moves
    return None, None


def extract_sokoban_answer(completion: str) -> str | None:
    """Extract normalized Sokoban moves from answer tags or after the first #### marker.

    Thin wrapper over find_sokoban_answer_end_and_moves — the single source of truth for
    answer extraction — so the training trim/score path and every eval path score identically
    by construction (they used to disagree on tag-vs-marker precedence and on spaced/word-form
    '####' answers, making the trained reward diverge from the eval reward)."""
    _, moves = find_sokoban_answer_end_and_moves(completion)
    return moves


@lru_cache(maxsize=1)
def _sokoban_scorer():
    return reasoning_gym.create_dataset("sokoban", size=1, seed=0)


def score_sokoban_moves(moves: str | None, entry: dict[str, Any]) -> float:
    return float(_sokoban_scorer().score_answer(answer=moves, entry=entry))


def _validate_sokoban_record(record: Any, *, path: Path, line_no: int) -> dict[str, Any]:
    prefix = f"{path}:{line_no}"
    if not isinstance(record, dict):
        raise ValueError(f"{prefix}: expected a JSON object")
    question = record.get("question")
    answer = record.get("answer")
    metadata = record.get("metadata")
    if not isinstance(question, str) or not question:
        raise ValueError(f"{prefix}: field 'question' must be a non-empty string")
    if not isinstance(answer, str) or not answer:
        raise ValueError(f"{prefix}: field 'answer' must be a non-empty string")
    if not isinstance(metadata, dict):
        raise ValueError(f"{prefix}: field 'metadata' must be an object")
    gamestr = metadata.get("gamestr")
    if not isinstance(gamestr, str) or not gamestr.strip():
        raise ValueError(f"{prefix}: field 'metadata.gamestr' must be a non-empty string")
    return {"question": question, "answer": answer, "metadata": metadata}


class FixedSokobanDataset:
    """Fixed JSONL Sokoban dataset with Reasoning Gym reward semantics."""

    def __init__(self, examples: list[dict[str, Any]], *, path: Path, split_name: str):
        if not examples:
            raise ValueError(f"{path}: {split_name} dataset is empty")
        self.examples = examples
        self.path = path
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = dict(self.examples[index])
        entry["metadata"] = dict(entry["metadata"])
        entry["messages"] = [
            {"role": "user", "content": entry["question"]},
            {"role": "assistant", "content": entry["answer"]},
        ]
        return entry

    @property
    def eval_type(self) -> str:
        return "generative"

    def reward(self, conversation: dict[str, Any], assistant_response: str) -> float:
        if not isinstance(assistant_response, str):
            raise TypeError("assistant_response must be a string")
        moves = extract_sokoban_answer(assistant_response)
        return score_sokoban_moves(moves, conversation)

    def evaluate(self, conversation: dict[str, Any], assistant_response: str) -> int:
        return int(self.reward(conversation, assistant_response) == 1.0)


def load_sokoban_jsonl_dataset(
    path: Path | str,
    *,
    split_name: str,
    verify_gold: bool = False,
) -> FixedSokobanDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{split_name} data file does not exist: {path}")
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank lines are not valid JSONL records")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            examples.append(_validate_sokoban_record(record, path=path, line_no=line_no))

    dataset = FixedSokobanDataset(examples, path=path, split_name=split_name)
    if verify_gold:
        failures: list[str] = []
        for idx, entry in enumerate(dataset.examples):
            moves = normalize_sokoban_moves(entry["answer"])
            if moves is None or score_sokoban_moves(moves, entry) != 1.0:
                failures.append(f"{path}:{idx + 1}")
                if len(failures) >= 5:
                    break
        if failures:
            raise ValueError(
                f"{split_name} dataset has gold answers that do not solve their boards: "
                + ", ".join(failures)
            )
    return dataset


# ============================ VLLM GENERATION CHILD (rollout + in-loop-eval producer) ============================
# vLLM scheduler/engine knobs plumbed verbatim from the CLI (`--vllm-*` for rollout generation,
# `--eval-vllm-*` for the standalone eval engine) into AsyncEngineArgs. None means "preserve
# vLLM's default" everywhere along the way. Adding a knob = one entry here + one add_argument in
# _add_vllm_tuning_args (both flag families pick it up automatically).
VLLM_TUNE_KEYS = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_num_scheduled_tokens",
    "max_num_partial_prefills",
    "max_long_partial_prefills",
    "long_prefill_token_threshold",
    "enable_chunked_prefill",
    "scheduler_reserve_full_isl",
    "stream_interval",
    "performance_mode",
)


def vllm_tuning_from_args(args: argparse.Namespace, prefix: str = "") -> dict[str, Any]:
    """Collect one --[eval-]vllm-* flag family into build_async_engine tuning kwargs."""
    return {key: getattr(args, f"{prefix}vllm_{key}") for key in VLLM_TUNE_KEYS}


def build_async_engine(
    model: str,
    num_dp: int,
    *,
    dtype: str = "bfloat16",
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    seed: int = 0,
    enforce_eager: bool = False,
    **tuning,
):
    """Construct the in-process vLLM AsyncLLM with native NCCL weight transfer enabled.

    `tuning` accepts exactly the VLLM_TUNE_KEYS knobs; None values are dropped so vLLM's own
    defaults apply."""
    unknown = set(tuning) - set(VLLM_TUNE_KEYS)
    if unknown:
        raise TypeError(f"unknown vLLM tuning kwargs: {sorted(unknown)}")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    from vllm.config import WeightTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_kwargs = {
        "model": model,
        "tensor_parallel_size": 1,
        "data_parallel_size": num_dp,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "enforce_eager": enforce_eager,
        "distributed_executor_backend": None,
        "max_model_len": max_model_len,
        "enable_log_requests": False,
        "seed": seed,
        "weight_transfer_config": WeightTransferConfig(backend="nccl"),
        # CISPO behavior logprobs must be PRE-temperature/top-p (the trainer recomputes
        # log pi at temperature 1.0, so the IS ratio is only consistent against raw
        # logprobs). This is vLLM's current default; pin it so an upstream default flip
        # to processed_logprobs can't silently bake a temperature offset into every IS
        # weight and the mismatch/ESS diagnostics.
        "logprobs_mode": "raw_logprobs",
    }
    engine_kwargs.update({key: value for key, value in tuning.items() if value is not None})
    engine_args = AsyncEngineArgs(**engine_kwargs)
    return AsyncLLM.from_engine_args(engine_args)


def extract_sampled_logprobs(completion_output) -> list[float]:
    """Per-token logprob of the sampled token for vLLM completion outputs."""
    co = completion_output
    if co.logprobs is None:
        return []
    return [co.logprobs[i][tok_id].logprob for i, tok_id in enumerate(co.token_ids)]


async def _vllm_generate(engine, prompt_token_ids, n, *, temperature, top_p, top_k, max_tokens, logprobs, seed):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    sp = SamplingParams(
        n=n, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens,
        logprobs=logprobs, seed=seed, output_kind=RequestOutputKind.FINAL_ONLY,
    )
    final = None
    async for out in engine.generate(TokensPrompt(prompt_token_ids=list(prompt_token_ids)), sp, str(uuid.uuid4())):
        final = out
    assert final is not None and final.finished
    # vLLM 0.22 can occasionally surface a None/empty completion in final.outputs (e.g. an aborted
    # sample); drop them so one bad sample can't crash the whole run.
    outputs = [co for co in final.outputs if co is not None and co.token_ids is not None]
    dropped = len(final.outputs) - len(outputs)
    if dropped:
        print(f"[generate_group] dropped {dropped}/{len(final.outputs)} None/empty vLLM completion(s)", flush=True)
    return outputs


def make_interrupt_config(
    marker_ids: list[int],
    answer_max_tokens: int,
    max_model_len: int,
    *,
    base_temperature: float,
    base_top_p: float,
    base_top_k: int,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    phase1_min_tokens: int | None = None,
    phase1_max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build the sampling['interrupt'] payload consumed by generate_group. The forced-answer
    continuation inherits the base sampling params unless an explicit override is given; the
    training / in-loop-eval / standalone-eval paths all build their config through here so the
    inherit-unless-overridden semantics cannot drift apart."""
    return {
        "marker_ids": list(marker_ids),
        "answer_max_tokens": answer_max_tokens,
        "max_model_len": max_model_len,
        "phase1_min_tokens": phase1_min_tokens,
        "phase1_max_tokens": phase1_max_tokens,
        "temperature": base_temperature if temperature is None else temperature,
        "top_p": base_top_p if top_p is None else top_p,
        "top_k": base_top_k if top_k is None else top_k,
    }


async def generate_group(engine, prompt_token_ids: list[int], num_samples: int, sampling: dict) -> list[dict]:
    """Generate num_samples completions for one prompt.

    Each returned dict: token_ids, logprobs (one per token), finish_reason, and loss_mask (one bool
    per token; False marks injected interruption-marker tokens that must NOT carry policy gradient).
    With sampling['interrupt'] set, any sample that hits the token cap while still thinking is
    continued once more after an injected end-of-thinking marker, so it emits a scoreable answer
    (ScaleRL A.10 interruption-based length control) instead of hard-truncating to reward 0."""
    temperature = sampling.get("temperature", 0.8)
    top_p = sampling.get("top_p", 1.0)
    top_k = sampling.get("top_k", 0)
    logprobs = sampling.get("logprobs", 0)
    seed = sampling.get("seed")
    max_tokens = int(sampling.get("max_tokens", 1024))
    interrupt = sampling.get("interrupt")
    if interrupt:
        phase1_min = interrupt.get("phase1_min_tokens")
        phase1_max = interrupt.get("phase1_max_tokens")
        if phase1_min is not None or phase1_max is not None:
            lo = int(phase1_min if phase1_min is not None else phase1_max)
            hi = int(phase1_max if phase1_max is not None else max_tokens)
            max_tokens = random.randint(lo, hi) if hi > lo else lo
    outputs = await _vllm_generate(
        engine, prompt_token_ids, num_samples,
        temperature=temperature, top_p=top_p, top_k=top_k,
        max_tokens=max_tokens, logprobs=logprobs, seed=seed,
    )

    results: list[dict] = []
    pending: list[tuple[int, list[int], list[float]]] = []  # (result index, phase-1 tokens, phase-1 logprobs)
    for co in outputs:
        toks = list(co.token_ids)
        lps = extract_sampled_logprobs(co)
        if interrupt and co.finish_reason == "length":
            # Still thinking at the cap: queue a forced-answer continuation. Placeholder filled below.
            pending.append((len(results), toks, lps))
            results.append(None)
        else:
            # loss_mask is attached only when interruption is active; absent => None downstream
            # (every generated token trainable), keeping the non-interruption path byte-identical.
            res = {"token_ids": toks, "logprobs": lps, "finish_reason": co.finish_reason}
            if interrupt:
                res["loss_mask"] = [True] * len(toks)
            results.append(res)

    if interrupt and pending:
        marker = list(interrupt["marker_ids"])
        answer_max = int(interrupt["answer_max_tokens"])
        max_model_len = int(interrupt.get("max_model_len", 0))
        answer_temperature = float(interrupt.get("temperature", temperature))
        answer_top_p = float(interrupt.get("top_p", top_p))
        answer_top_k = int(interrupt.get("top_k", top_k))

        async def _continue(idx: int, phase1: list[int], phase1_lps: list[float]):
            # Clamp the forced-answer budget so the phase-2 prompt never exceeds max_model_len
            # (vLLM would otherwise abort the request and crash the run). If there's no room, keep
            # phase-1 as-is (a length-truncated rollout, same as without interruption).
            budget = answer_max
            if max_model_len:
                budget = min(answer_max, max_model_len - len(prompt_token_ids) - len(phase1) - len(marker))
            if budget <= 0:
                results[idx] = {"token_ids": phase1, "logprobs": phase1_lps,
                                "finish_reason": "length", "loss_mask": [True] * len(phase1)}
                return
            cont = await _vllm_generate(
                engine, list(prompt_token_ids) + phase1 + marker, 1,
                temperature=answer_temperature, top_p=answer_top_p, top_k=answer_top_k,
                max_tokens=budget, logprobs=logprobs, seed=seed,
            )
            ans_toks = list(cont[0].token_ids) if cont else []
            ans_lps = extract_sampled_logprobs(cont[0]) if cont else []
            ans_fr = cont[0].finish_reason if cont else "length"
            results[idx] = {
                # Marker tokens are injected context: kept in the sequence (the answer was generated
                # conditioned on them) but flagged non-trainable via loss_mask. Their logprob slot is a
                # placeholder (0.0) — the trainer masks those positions out entirely.
                "token_ids": phase1 + marker + ans_toks,
                "logprobs": phase1_lps + [0.0] * len(marker) + ans_lps,
                "finish_reason": ans_fr,
                "loss_mask": [True] * len(phase1) + [False] * len(marker) + [True] * len(ans_toks),
            }

        await asyncio.gather(*[_continue(i, p1, lp) for (i, p1, lp) in pending])

    return results


def _safe_get(q):
    """Non-throwing queue get with a short timeout; returns None on empty."""
    try:
        return q.get(timeout=0.1)
    except queue.Empty:
        return None


async def _drain_prompts(engine, prompt_q, result_q, state, inflight: int):
    """Keep up to `inflight` vLLM generations running."""
    pending: set = set()
    deferred_req = None

    async def one(req, launch_weight_version: int):
        samples = await generate_group(engine, req["prompt_token_ids"], req["num_samples"], req["sampling"])
        result_q.put(
            {
                "type": MSG_ROLLOUT,
                "puzzle_id": req["puzzle_id"],
                "weight_version": launch_weight_version,
                "samples": samples,
            }
        )

    while not state["stop"]:
        while not state["paused"] and len(pending) < inflight:
            if deferred_req is None:
                req = await asyncio.to_thread(_safe_get, prompt_q)
            else:
                req = deferred_req
                deferred_req = None
            if req is None:
                break
            if state["paused"]:
                deferred_req = req
                break
            if req.get("type") == MSG_GENERATE:
                pending.add(asyncio.create_task(one(req, state["weight_version"])))
        if pending:
            done, pending = await asyncio.wait(pending, timeout=0.05, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    import traceback
                    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    result_q.put({"type": MSG_ERROR, "msg": f"generation failed: {exc!r}\n{tb}"})
        else:
            await asyncio.sleep(0.02)


async def _handle_control(engine, control_q, status_q, state):
    from vllm.distributed.weight_transfer.base import WeightTransferInitRequest, WeightTransferUpdateRequest
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLWeightTransferInitInfo,
        NCCLWeightTransferUpdateInfo,
    )

    while not state["stop"]:
        msg = await asyncio.to_thread(_safe_get, control_q)
        if msg is None:
            continue
        if msg["type"] == MSG_SHUTDOWN:
            state["stop"] = True
            break
        if msg["type"] == MSG_INIT_WEIGHTS:
            await engine.init_weight_transfer_engine(
                WeightTransferInitRequest(
                    init_info=asdict(
                        NCCLWeightTransferInitInfo(
                            master_address=msg["master_address"],
                            master_port=msg["master_port"],
                            rank_offset=1,
                            world_size=msg["world_size"],
                        )
                    )
                )
            )
            status_q.put({"type": MSG_ENGINE_READY})
        elif msg["type"] == MSG_SYNC_WEIGHTS:
            state["paused"] = True
            await engine.pause_generation(mode="keep")
            await engine.start_weight_update(is_checkpoint_format=True)
            await engine.update_weights(
                WeightTransferUpdateRequest(
                    update_info=asdict(
                        NCCLWeightTransferUpdateInfo(
                            names=msg["names"],
                            dtype_names=msg["dtype_names"],
                            shapes=msg["shapes"],
                            packed=True,
                        )
                    )
                )
            )
            await engine.finish_weight_update()
            await engine.resume_generation()
            state["weight_version"] = msg["version"]
            state["paused"] = False
            status_q.put({"type": MSG_WEIGHTS_READY, "version": msg["version"]})


async def _child_main_async(cfg, prompt_q, result_q, control_q, status_q):
    engine = build_async_engine(
        cfg["model"],
        cfg["num_dp"],
        dtype=cfg["dtype"],
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        seed=cfg["seed"],
        enforce_eager=cfg.get("enforce_eager", False),
        **{key: cfg.get(key) for key in VLLM_TUNE_KEYS},
    )
    state = {"weight_version": 0, "paused": False, "stop": False}
    status_q.put({"type": MSG_ENGINE_READY})
    try:
        await asyncio.gather(
            _drain_prompts(engine, prompt_q, result_q, state, cfg["inflight"]),
            _handle_control(engine, control_q, status_q, state),
        )
    finally:
        engine.shutdown()


def engine_child_main(cfg, prompt_q, result_q, control_q, status_q):
    """Spawn target for the vLLM generator child."""
    random.seed(cfg["seed"])
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["visible_gpus"]
    for key in list(os.environ):
        if key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "GROUP_RANK",
            "GROUP_WORLD_SIZE",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "ROLE_NAME",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ) or key.startswith(("TORCHELASTIC_", "TORCH_NCCL_")):
            os.environ.pop(key, None)
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    os.environ.setdefault("VLLM_DP_MASTER_IP", "127.0.0.1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        asyncio.run(_child_main_async(cfg, prompt_q, result_q, control_q, status_q))
    except Exception as exc:
        status_q.put({"type": MSG_ERROR, "msg": f"engine child crashed: {exc!r}"})
        raise


def _await_status(status_q, expected_type, timeout: float = 600.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = status_q.get(timeout=1.0)
        except Exception:
            continue
        if msg["type"] == "error":
            raise RuntimeError(f"engine child error: {msg['msg']}")
        if msg["type"] == expected_type:
            return msg
    raise TimeoutError(f"timed out waiting for child status {expected_type!r}")


def _start_process_with_cuda_visible_devices(process, visible_gpus: str) -> None:
    """Start a spawn child with CVD already present in its inherited environment."""
    old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
    try:
        process.start()
    finally:
        if old_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd


# ============================ TRAINER DATA STRUCTURES & DISTRIBUTED PLUMBING ============================
@dataclass
class RolloutExample:
    """One generated sequence carried through the async pipeline.

    Per-sequence: carries the vLLM behavior log-probs and the trainer weight
    version that produced it, for CISPO off-policy correction and staleness control.
    """
    sequence: torch.Tensor          # full token ids (prompt + generated), 1-D long
    prefix_length: int              # number of prompt tokens
    reward: float
    advantage: float
    behavior_logprobs: list[float]  # one per generated token; len == sequence.numel() - prefix_length
    weight_version: int
    puzzle_id: int
    completion: str = ""
    loss_mask: list[bool] | None = None
    group_token_count: int = 0


@dataclass(frozen=True)
class PromptCacheEntry:
    """Pre-tokenized fixed training prompt used by rank 0's rollout enqueuer."""

    conversation: dict[str, Any]
    prompt_ids: torch.Tensor
    prompt_token_ids: list[int]
    prefix_length: int


@dataclass
class ProcessedRolloutSample:
    """One vLLM sample after trim/decode/extract/score processing."""

    sequence: torch.Tensor
    behavior_logprobs: list[float]
    loss_mask: list[bool] | None
    completion: str
    moves: str | None
    reward: float
    finish_reason: str | None


@dataclass(frozen=True)
class ModelPerfInfo:
    parameter_count: int
    parameter_bytes: int


def collect_model_perf_info(model: torch.nn.Module) -> ModelPerfInfo:
    """Static model-size counters used for coarse roofline-style estimates."""
    return ModelPerfInfo(
        parameter_count=sum(param.numel() for param in model.parameters()),
        parameter_bytes=sum(param.numel() * param.element_size() for param in model.parameters()),
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def estimate_step_perf_metrics(
    *,
    model_info: ModelPerfInfo,
    train_padded_tokens: float,
    train_forward_backward_passes: float,
    rollout_prefill_tokens: float,
    rollout_decode_tokens: float,
    rollout_forward_passes: float,
    train_seconds: float,
    rollout_seconds: float,
) -> dict[str, float]:
    """Estimate dense-model FLOPs and weight-byte arithmetic intensity.

    These are intentionally simple counters: dense training is approximated as
    6 * params * tokens, inference as 2 * params * tokens. The byte denominator
    counts parameter bytes per forward/backward or rollout forward pass, so this
    is a weight-streaming arithmetic-intensity estimate rather than a profiler
    measurement of all HBM traffic.
    """
    params = float(model_info.parameter_count)
    param_bytes = float(model_info.parameter_bytes)
    rollout_model_tokens = rollout_prefill_tokens + rollout_decode_tokens

    train_flops = 6.0 * params * train_padded_tokens
    rollout_flops = 2.0 * params * rollout_model_tokens
    total_flops = train_flops + rollout_flops

    train_weight_bytes = param_bytes * train_forward_backward_passes
    rollout_weight_bytes = param_bytes * rollout_forward_passes
    total_weight_bytes = train_weight_bytes + rollout_weight_bytes
    total_seconds = train_seconds + rollout_seconds

    return {
        "perf_est/model_params": params,
        "perf_est/model_param_bytes": param_bytes,
        "perf_est/train_padded_tokens": train_padded_tokens,
        "perf_est/train_forward_backward_passes": train_forward_backward_passes,
        "perf_est/rollout_prefill_tokens": rollout_prefill_tokens,
        "perf_est/rollout_decode_tokens": rollout_decode_tokens,
        "perf_est/rollout_model_tokens": rollout_model_tokens,
        "perf_est/rollout_forward_passes": rollout_forward_passes,
        "perf_est/train_flops": train_flops,
        "perf_est/rollout_flops": rollout_flops,
        "perf_est/total_flops": total_flops,
        "perf_est/train_tflops_per_s": _safe_div(train_flops, train_seconds * 1e12),
        "perf_est/rollout_tflops_per_s": _safe_div(rollout_flops, rollout_seconds * 1e12),
        "perf_est/step_tflops_per_s": _safe_div(total_flops, total_seconds * 1e12),
        "perf_est/train_tokens_per_s": _safe_div(train_padded_tokens, train_seconds),
        "perf_est/rollout_model_tokens_per_s": _safe_div(rollout_model_tokens, rollout_seconds),
        "perf_est/train_ai_flop_per_weight_byte": _safe_div(train_flops, train_weight_bytes),
        "perf_est/rollout_ai_flop_per_weight_byte": _safe_div(rollout_flops, rollout_weight_bytes),
        "perf_est/step_ai_flop_per_weight_byte": _safe_div(total_flops, total_weight_bytes),
    }


def print0(message: str = "", **kwargs) -> None:
    if int(os.environ.get("RANK", 0)) == 0:
        kwargs.setdefault("flush", True)
        print(message, **kwargs)


def is_ddp_requested(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    present = [name for name in required if name in env]
    if present and len(present) != len(required):
        missing = ", ".join(name for name in required if name not in env)
        raise ValueError(f"Incomplete DDP environment; missing {missing}")
    return len(present) == len(required)


def get_dist_info(env: Mapping[str, str] | None = None) -> tuple[bool, int, int, int]:
    env = os.environ if env is None else env
    if not is_ddp_requested(env):
        return False, 0, 0, 1

    ddp_rank = int(env["RANK"])
    ddp_local_rank = int(env["LOCAL_RANK"])
    ddp_world_size = int(env["WORLD_SIZE"])
    if ddp_world_size < 1:
        raise ValueError("WORLD_SIZE must be at least 1")
    if ddp_rank < 0 or ddp_rank >= ddp_world_size:
        raise ValueError(f"RANK must be in [0, WORLD_SIZE), got {ddp_rank}/{ddp_world_size}")
    if ddp_local_rank < 0:
        raise ValueError("LOCAL_RANK must be non-negative")
    return True, ddp_rank, ddp_local_rank, ddp_world_size


def build_optimizer(parameters: Iterator[torch.nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer != "adamw":
        raise ValueError(f"Unsupported optimizer {args.optimizer!r}; only 'adamw' is implemented")
    params = list(parameters)
    # fused=True single-kernels the update over the ~64GB of fp32 param/grad/state traffic
    # (~0.5-1s/step at 4B params) instead of the multi-kernel foreach path. Same update math
    # (not bitwise). CUDA-only, so fall back to the default path on CPU (tests / --device cpu).
    use_fused = bool(params) and all(p.is_cuda for p in params)
    return torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
        fused=use_fused if use_fused else None,
    )


def get_lr_multiplier(
    step: int,
    num_steps: int,
    init_lr_frac: float = 1.0,
    warmup_steps: int = 0,
    schedule: str = "linear",
    min_lr_frac: float = 0.0,
    lr_decay_steps: int | None = None,
    lr_decay_start: int | None = None,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        if warmup_steps == 1:
            return 1.0
        progress = step / (warmup_steps - 1)
        return init_lr_frac + (1.0 - init_lr_frac) * progress
    if schedule == "constant":
        return 1.0
    if schedule != "linear":
        raise ValueError(f"Unsupported LR schedule: {schedule}")
    # Linear decay from peak (1.0, at the end of warmup) down to the floor `min_lr_frac`, reached at
    # absolute step `lr_decay_steps` (defaults to num_steps), then HELD at the floor for the rest of
    # the run. With min_lr_frac=0.0 and lr_decay_steps=None this is the legacy decay-to-zero schedule.
    decay_end = lr_decay_steps if lr_decay_steps is not None else num_steps
    if warmup_steps == 0:
        if num_steps <= 1:
            return 1.0
        if step == 0:
            return init_lr_frac
        decay_span = max(1, decay_end)
        decay_progress = min(1.0, max(0.0, (step - 1) / decay_span))
        return min_lr_frac + (1.0 - min_lr_frac) * (1.0 - decay_progress)
    # Optional HOLD at peak until `lr_decay_start` (defaults to the end of warmup), then linear decay.
    decay_start = lr_decay_start if lr_decay_start is not None else warmup_steps
    if step < decay_start:
        return 1.0
    decay_span = max(1, decay_end - decay_start)
    decay_progress = min(1.0, max(0.0, (step - decay_start) / decay_span))
    return min_lr_frac + (1.0 - min_lr_frac) * (1.0 - decay_progress)


def make_pad_example(pad_token_id: int) -> "RolloutExample":
    """A RolloutExample that pads a step batch to a multiple of the trainer world size, so
    every rank gets equal microbatches and none stalls in the all-reduce. It contributes zero
    to both loss numerator and token-count denominator: advantage 0.0 zeroes the gradient,
    prefix_length == length means no generated tokens, and empty behavior_logprobs keeps it out
    of both counts (see local_and_global_counts). The 2-token sequence satisfies
    build_rl_batch_varprefix_cpu's max_length >= 2 guard.
    """
    return RolloutExample(
        sequence=torch.tensor([pad_token_id, pad_token_id], dtype=torch.long),
        prefix_length=2,
        reward=0.0,
        advantage=0.0,
        behavior_logprobs=[],
        weight_version=0,
        puzzle_id=-1,
        completion="",
    )


def strip_rollout_completion(example: "RolloutExample") -> "RolloutExample":
    """Return a training-only copy of a rollout without the decoded completion payload."""
    return example if example.completion == "" else replace(example, completion="")


def strip_rollout_completions(examples: list["RolloutExample"]) -> list["RolloutExample"]:
    """Drop decoded strings from examples that are only needed for training/scatter payloads."""
    return [strip_rollout_completion(example) for example in examples]


def sort_and_shard_for_microbatching(
    examples: list["RolloutExample"],
    advantages: list[float],
    world_size: int,
    pad_example: "RolloutExample",
) -> tuple[list[list["RolloutExample"]], list[list[float]]]:
    """Length-sort the full step batch and deal it across ranks for minimal padding + balanced load.

    The trained advantage is the batch-normalized adv_all value, NOT example.advantage (which
    still holds the pre-normalization group-centered value), and the train loop indexes it
    positionally — so (example, advantage) travel as joint pairs here and cannot desync.

    1. Sort pairs by sequence length DESCENDING: adjacent device_batch_size slices then pad to
       near-equal lengths (cuts the ~9% pad-token waste of arrival-order pairing to ~2%), and
       the 2-token pad examples land at the short end next to the shortest real sequences.
    2. Snake-deal the sorted list across ranks (even round left-to-right, odd round right-to-
       left) so every rank gets the same length PROFILE — a contiguous split of a sorted list
       would concentrate the long sequences on one rank (~11% imbalance) and the Phase-E/F
       collectives sync on the slowest rank.
    3. Pad every rank to the same count with `pad_example` (zero gradient by construction, see
       make_pad_example) so no rank stalls the grad all-reduce.

    Gradient-identical to the unsorted contiguous path up to fp summation order: every policy
    loss here (cispo/grpo/gspo) is per-sequence separable with step-GLOBAL normalizers (Phase D)
    — gspo's sequence ratio is computed within each row — so neither order, pairing, nor rank
    assignment changes the summed gradient.
    """
    if world_size < 1:
        raise ValueError("world_size must be at least 1")
    if len(examples) != len(advantages):
        raise ValueError("examples and advantages must align")
    if pad_example.advantage != 0.0:
        raise ValueError("pad_example must have advantage == 0.0 (zero gradient)")
    pairs = sorted(zip(examples, advantages),
                   key=lambda pair: int(pair[0].sequence.numel()), reverse=True)
    rank_pairs: list[list[tuple["RolloutExample", float]]] = [[] for _ in range(world_size)]
    for index, pair in enumerate(pairs):
        round_index, position = divmod(index, world_size)
        rank = position if round_index % 2 == 0 else world_size - 1 - position
        rank_pairs[rank].append(pair)
    per_rank = max(len(rp) for rp in rank_pairs)
    shards: list[list["RolloutExample"]] = []
    adv_shards: list[list[float]] = []
    for rp in rank_pairs:
        padded = rp + [(pad_example, 0.0)] * (per_rank - len(rp))
        shards.append([example for example, _ in padded])
        adv_shards.append([advantage for _, advantage in padded])
    return shards, adv_shards


def local_and_global_counts(local_examples: list["RolloutExample"]) -> tuple[int, int]:
    """Per-shard (sample_count, token_count) for one trainer rank, counting only real
    examples. Pad examples (empty behavior_logprobs) add 0 to BOTH counts, so summing
    across ranks recovers the unpadded global counts and the loss stays byte-equivalent
    to the unpadded full-batch gradient in both normalization branches.
    """
    local_sample_count = sum(1 for e in local_examples if e.behavior_logprobs)
    # Count only gradient-carrying tokens: injected interruption markers sit in behavior_logprobs as
    # placeholders but are masked out of the loss, so exclude them to keep the token normalizer and
    # the train-throughput counters honest (no-op when loss_mask is None, i.e. interruption off).
    local_token_count = sum(
        (sum(e.loss_mask) if e.loss_mask is not None else len(e.behavior_logprobs))
        for e in local_examples
    )
    return local_sample_count, local_token_count


def _scatter_payload(
    padded_batch: list["RolloutExample"],
    adv_padded: list[float] | None,
    abort: bool = False,
    num_prompts: int = 0,
) -> list:
    """The object rank 0 broadcasts to every trainer rank each step:
    [control dict with abort flag + prompt count, full padded step batch, full-batch advantages].

    Each rank reads element 0 first, then either raises (abort) or slices its shard.
    Putting the abort flag first gives a poison sentinel and a real scatter the same
    wire shape, so a rank-0 Phase-A failure unblocks workers in lockstep. num_prompts (the
    global count of trained prompt groups P) rides in the control dict for the ScaleRL
    prompt-level loss normalizer, which every rank needs but only rank 0 can compute.
    """
    return [{"abort": bool(abort), "num_prompts": int(num_prompts)}, padded_batch, adv_padded]


def _abort_payload() -> list:
    """Poison sentinel broadcast by rank 0 before re-raising a Phase-A/G error, so every
    worker raises in lockstep after the (uniform) broadcast completes (abort protocol)."""
    return _scatter_payload([], None, abort=True)


def _pipeline_init_status_payload(ok: bool, error: str | None = None) -> list:
    """Rank-0 init status broadcast before workers enter the pipeline step loop."""
    return [{"ok": bool(ok), "error": error}]


def _raise_for_pipeline_init_status(status: dict) -> None:
    if not status.get("ok", False):
        error = status.get("error") or "rank 0 pipeline initialization failed"
        raise RuntimeError(f"rank 0 pipeline initialization failed: {error}")


@dataclass(frozen=True)
class PipelineTrainerLayout:
    """Physical-GPU placement for the pipeline launch. The T trainer ranks occupy GPUs 0..T-1
    (each does set_device(LOCAL_RANK), so CUDA_VISIBLE_DEVICES must stay UNSET); the vLLM child
    takes the remaining GPUs T..T+M-1, exported to it via CUDA_VISIBLE_DEVICES=`visible_gpus`.
    """

    trainer_ranks: int
    visible_gpus: str


def pipeline_trainer_layout(world_size: int, vllm_dp: int) -> PipelineTrainerLayout:
    """Compute the trainer/vLLM physical-GPU split. `world_size` (torchrun WORLD_SIZE) is the
    trainer rank count T; the vLLM child gets the contiguous block GPUs T..T+M-1 after it. Total
    T+M must fit one 8-GPU node, so reject oversubscription rather than collide two procs on a GPU.
    """
    if world_size < 1:
        raise ValueError("world_size (trainer ranks) must be at least 1")
    if vllm_dp < 1:
        raise ValueError("vllm_dp (generator GPUs) must be at least 1")
    total = world_size + vllm_dp
    if total > NODE_GPUS:
        raise ValueError(
            f"trainer GPUs + vLLM generators must fit one {NODE_GPUS}-GPU node, got "
            f"{world_size} + {vllm_dp} = {total} > {NODE_GPUS}"
        )
    visible = ",".join(str(i) for i in range(world_size, world_size + vllm_dp))
    return PipelineTrainerLayout(trainer_ranks=world_size, visible_gpus=visible)


def compute_cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_grads_sum(model: torch.nn.Module, group=None) -> None:
    """Manual replacement for DDP's grad hooks (the trainer never DDP-wraps the model, since
    chunked_policy_loss reaches into model.model / model.lm_head directly). Each rank runs its
    per-shard loss with normalizer=1.0 over GLOBAL counts, then all_reduce(SUM) every grad to
    recover the exact single-GPU full-batch gradient.

    Zero-fill-before-all_reduce is MANDATORY: every rank must issue the collective for every
    parameter in the SAME order — even params with no local grad this step — or NCCL deadlocks
    on a non-uniform collective set. Assumes all params are trainable; a frozen param would
    need a requires_grad filter applied identically on every rank.
    """
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)


def assert_replica_sync(model: torch.nn.Module, step: int, group=None) -> None:
    """Debug guard: assert every trainer-rank replica is bit-identical after an optimizer step
    (identical all-reduced-SUM grads + identical AdamW state). Compares a deterministic
    double-precision checksum across the trainer group via all_reduce(MIN)/all_reduce(MAX) —
    MIN != MAX means divergence — over ALL params and again over the single synced
    `embed_tokens.weight` (the only param weight-synced to vLLM, so its divergence is critical).

    Load-bearing: this issues four extra collectives (2x MIN, 2x MAX) in a fixed order, so every
    rank MUST call it in lockstep — the caller gates it by `world_size > 1` (and the debug flag).
    Place it AFTER optimizer.step() and BEFORE the Phase-G barrier, never between rank 0's
    MSG_SYNC_WEIGHTS put and MSG_WEIGHTS_READY await. Raises RuntimeError on divergence.
    """
    device = next(model.parameters()).device

    def _checksum(t: torch.Tensor) -> torch.Tensor:
        # Deterministic double-precision scalar checksum on the model's device.
        lo = t.clone()
        hi = t.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=group)
        return lo, hi

    # --- (1) checksum over ALL params ---
    all_chk = torch.stack([p.detach().double().sum() for p in model.parameters()]).sum()
    lo_all, hi_all = _checksum(all_chk)
    if not torch.equal(lo_all, hi_all):
        raise RuntimeError(
            f"replica divergence at step {step}: checksum min={lo_all} max={hi_all}"
        )

    # --- (2) the single synced param specifically (tied embed_tokens.weight) ---
    synced = None
    for name, p in model.named_parameters():
        if name.endswith("embed_tokens.weight"):
            synced = p
            break
    if synced is None:
        # Fall back to the first param (keeps the collective set uniform across ranks).
        synced = next(model.parameters())
    synced_chk = synced.detach().double().sum().to(device)
    lo_s, hi_s = _checksum(synced_chk)
    if not torch.equal(lo_s, hi_s):
        raise RuntimeError(
            f"synced-param (embed_tokens.weight) divergence at step {step}: "
            f"checksum min={lo_s} max={hi_s}"
        )


def set_seed(seed: int, device: torch.device | str | None = None) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


# ============================ PROMPT ENCODING & ROLLOUT POST-PROCESSING ============================
def as_1d_token_tensor(encoded) -> torch.Tensor:
    if isinstance(encoded, torch.Tensor):
        if encoded.ndim == 2:
            encoded = encoded[0]
        return encoded.to(dtype=torch.long)
    if isinstance(encoded, dict):
        return as_1d_token_tensor(encoded["input_ids"])
    if hasattr(encoded, "input_ids"):
        return as_1d_token_tensor(encoded.input_ids)
    if hasattr(encoded, "ids"):
        return torch.tensor(encoded.ids, dtype=torch.long)
    if isinstance(encoded, (list, tuple)):
        if not encoded:
            raise ValueError("Tokenizer returned an empty token sequence")
        first = encoded[0]
        if isinstance(first, int):
            return torch.tensor(encoded, dtype=torch.long)
        return as_1d_token_tensor(first)
    raise TypeError(f"Unsupported tokenizer output type: {type(encoded)}")


def encode_prompt(
    tokenizer,
    question: str,
    enable_thinking: bool = True,
) -> torch.Tensor:
    prompt = build_sokoban_prompt(question)
    messages = [{"role": "user", "content": prompt}]

    if getattr(tokenizer, "chat_template", None) is not None:
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=enable_thinking,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        return as_1d_token_tensor(encoded)

    fallback_text = f"User: {prompt}\nAssistant:"
    encoded = tokenizer(fallback_text, return_tensors="pt", add_special_tokens=True)
    return as_1d_token_tensor(encoded)


def build_prompt_cache(
    task: FixedSokobanDataset,
    tokenizer,
    enable_thinking: bool = True,
) -> list[PromptCacheEntry]:
    """Pre-tokenize every fixed training prompt once for the rank-0 rollout producer."""
    cache: list[PromptCacheEntry] = []
    for idx in range(len(task)):
        conversation = task[idx]
        prompt_ids = encode_prompt(tokenizer, conversation["question"], enable_thinking)
        prompt_token_ids = prompt_ids.tolist()
        cache.append(
            PromptCacheEntry(
                conversation=conversation,
                prompt_ids=prompt_ids,
                prompt_token_ids=prompt_token_ids,
                prefix_length=int(prompt_ids.numel()),
            )
        )
    if not cache:
        raise ValueError("Cannot build a prompt cache for an empty task")
    return cache


def decode_completion(tokenizer, sequence: torch.Tensor, prefix_length: int) -> str:
    generated = sequence[prefix_length:]
    return tokenizer.decode(generated.tolist(), skip_special_tokens=True)


def trim_generated_with_logprobs_and_text(
    tokenizer,
    sequence: torch.Tensor,
    prefix_length: int,
    behavior_logprobs: list[float],
    loss_mask: list[bool] | None = None,
) -> tuple[torch.Tensor, list[float], list[bool] | None, str, str | None]:
    """Trim after the first parseable answer and return the decoded kept completion text.

    This keeps rollout processing from decoding the final sequence a second time after trimming.
    The binary search still decodes candidate prefixes when trimming is needed, matching the
    previous token-boundary behavior.
    """
    generated_ids = sequence[prefix_length:].tolist()
    if not generated_ids:
        return sequence, behavior_logprobs, loss_mask, "", None
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    answer_end, moves = find_sokoban_answer_end_and_moves(full_text)
    if answer_end is None:
        return sequence, behavior_logprobs, loss_mask, full_text, None
    target_text = full_text[:answer_end]
    target_len = len(target_text)

    # Decode is prefix-preserving on standard tokenizers: len(decode([:end])) is non-decreasing
    # in `end`, so binary-search the smallest token end that covers the answer marker.
    lo, hi = 1, len(generated_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        text = tokenizer.decode(generated_ids[:mid], skip_special_tokens=True)
        if len(text) >= target_len:
            hi = mid
        else:
            lo = mid + 1
    final_text = tokenizer.decode(generated_ids[:lo], skip_special_tokens=True)
    if final_text[:target_len] != target_text:
        return sequence, behavior_logprobs, loss_mask, full_text, moves

    trimmed = sequence[: prefix_length + lo]
    kept_gen = int(trimmed.numel()) - prefix_length
    trimmed_mask = loss_mask[:kept_gen] if loss_mask is not None else None
    return trimmed, behavior_logprobs[:kept_gen], trimmed_mask, final_text, moves


def process_rollout_sample(
    tokenizer,
    prompt_ids: torch.Tensor,
    prefix_length: int,
    sample: dict[str, Any],
    conversation: dict[str, Any],
    *,
    trim_after_answer: bool,
) -> ProcessedRolloutSample:
    """Convert one vLLM sample into trainable fields, extracting the Sokoban answer once."""
    behavior_logprobs = list(sample["logprobs"])
    loss_mask = sample.get("loss_mask")
    if loss_mask is not None:
        loss_mask = list(loss_mask)
    sequence = torch.cat([prompt_ids, torch.tensor(sample["token_ids"], dtype=torch.long)])
    if trim_after_answer:
        sequence, behavior_logprobs, loss_mask, completion, moves = trim_generated_with_logprobs_and_text(
            tokenizer,
            sequence,
            prefix_length,
            behavior_logprobs,
            loss_mask,
        )
    else:
        completion = decode_completion(tokenizer, sequence, prefix_length)
        moves = extract_sokoban_answer(completion)
    reward = score_sokoban_moves(moves, conversation)
    return ProcessedRolloutSample(
        sequence=sequence,
        behavior_logprobs=behavior_logprobs,
        loss_mask=loss_mask,
        completion=completion,
        moves=moves,
        reward=reward,
        finish_reason=sample.get("finish_reason"),
    )


# ============================ MICROBATCH TENSORIZATION & CISPO LOSS ============================
def build_rl_batch_varprefix_cpu(sequences, prefixes, pad_token_id, masks=None, pin_memory=False):
    """Pad/concat + mask construction for one RL microbatch, with NO CUDA calls beyond
    optional pinned allocation, so a prefetch thread can build the NEXT microbatch while
    the current one's fwd/bwd runs; the train loop then issues one non_blocking H2D copy
    per tensor instead of per-row pageable copies."""
    if not sequences:
        raise ValueError("Cannot create an RL batch from zero sequences")
    if len(sequences) != len(prefixes):
        raise ValueError("sequences and prefixes must align")
    if masks is not None and len(masks) != len(sequences):
        raise ValueError("masks must align with sequences")
    max_length = max(int(seq.numel()) for seq in sequences)
    if max_length < 2:
        raise ValueError("Sequences must contain at least two tokens")
    ids = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long,
                     pin_memory=pin_memory)
    real = torch.zeros((len(sequences), max_length), dtype=torch.bool, pin_memory=pin_memory)
    gen = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    for row, (seq, pfx) in enumerate(zip(sequences, prefixes)):
        seq = seq.to(device="cpu", dtype=torch.long)
        n = int(seq.numel())
        ids[row, :n] = seq
        real[row, :n] = True
        if n > pfx:
            gen[row, pfx:n] = True
            # Drop injected (interruption-marker) positions from the training targets: present in
            # `real` (attended to as context) but excluded from `gen`/labels so they carry no gradient.
            m = masks[row] if masks is not None else None
            if m is not None:
                if len(m) != n - pfx:
                    raise ValueError(f"row {row}: loss_mask len {len(m)} != generated len {n - pfx}")
                gen[row, pfx:n] &= torch.tensor(m, dtype=torch.bool)
    input_ids = ids[:, :-1]
    # ids[:, :-1] / real[:, :-1] are views of pinned storage and stay pinned; a plain .clone()
    # for labels would allocate UNPINNED memory and silently degrade its non_blocking H2D copy,
    # so copy into an explicitly (optionally pinned) buffer instead.
    labels = torch.empty((len(sequences), max_length - 1), dtype=torch.long, pin_memory=pin_memory)
    labels.copy_(ids[:, 1:])
    labels.masked_fill_(~gen[:, 1:], IGNORE_INDEX)
    return input_ids, real[:, :-1], labels


def build_behavior_logprob_tensor_varprefix_cpu(sequences, behavior_logprobs, prefixes,
                                                pin_memory=False):
    """Behavior-logprob tensor for one RL microbatch (see build_rl_batch_varprefix_cpu)."""
    if not (len(sequences) == len(behavior_logprobs) == len(prefixes)):
        raise ValueError("sequences, behavior_logprobs, and prefixes must align")
    max_length = max(int(seq.numel()) for seq in sequences)
    out = torch.zeros((len(sequences), max_length - 1), dtype=torch.float32, pin_memory=pin_memory)
    for row, (seq, blp, pfx) in enumerate(zip(sequences, behavior_logprobs, prefixes)):
        gen_len = int(seq.numel()) - pfx
        if gen_len <= 0:
            continue
        if len(blp) != gen_len:
            raise ValueError(f"row {row}: {len(blp)} logprobs for {gen_len} generated tokens")
        start = pfx - 1
        out[row, start:start + gen_len] = torch.tensor(blp, dtype=torch.float32)
    return out


def tensorize_microbatch_cpu(microbatch, pad_token_id, pin_memory=False):
    """Build one microbatch's training tensors on host memory (optionally pinned).

    Runs on the tensorize prefetch thread: pure CPU tensor construction, no CUDA kernels and
    no dist.* collectives (the trainer NCCL group is single-threaded on the main thread; pinned
    allocation via cudaHostAlloc is thread-safe and not a collective)."""
    seqs = [e.sequence for e in microbatch]
    prefixes = [e.prefix_length for e in microbatch]
    masks = [e.loss_mask for e in microbatch]
    input_ids, attention_mask, labels = build_rl_batch_varprefix_cpu(
        seqs, prefixes, pad_token_id, masks=masks, pin_memory=pin_memory)
    beh = build_behavior_logprob_tensor_varprefix_cpu(
        seqs, [e.behavior_logprobs for e in microbatch], prefixes, pin_memory=pin_memory)
    return input_ids, attention_mask, labels, beh


def token_logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels != IGNORE_INDEX
    vocab_size = logits.size(-1)
    loss_logits = logits.float() if logits.dtype == torch.float16 else logits
    nll = F.cross_entropy(
        loss_logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).float()
    token_logprobs = -nll.view_as(labels)
    return token_logprobs.masked_fill(~valid, 0.0)


def chunked_token_logprobs(lm_head, hidden_states, labels, chunk_size=1024):
    """Per-token logprobs computed by applying `lm_head` to `hidden_states` in
    sequence-chunks under gradient checkpointing, so the full (T, vocab) logits are
    never materialized at once (backward recomputes one chunk's logits at a time).
    Equivalent to token_logprobs_from_logits(lm_head(hidden_states).float(), labels)."""
    import torch.utils.checkpoint as _ckpt
    T = hidden_states.size(1)
    parts = []
    for i in range(0, T, chunk_size):
        h = hidden_states[:, i:i + chunk_size, :]
        lab = labels[:, i:i + chunk_size]

        def _f(h_chunk, lab_chunk):
            # fp32 logits regardless of any outer autocast (ScaleRL's fp32-at-the-head): disable
            # autocast and upcast the hidden input so the head matmul runs in fp32. The trailing
            # .float() also normalizes the output for a bf16-weight head; keep both.
            with torch.autocast(device_type=h_chunk.device.type, enabled=False):
                logits = lm_head(h_chunk.float()).float()
            return token_logprobs_from_logits(logits, lab_chunk)

        if torch.is_grad_enabled() and hidden_states.requires_grad:
            parts.append(_ckpt.checkpoint(_f, h, lab, use_reentrant=False))
        else:
            parts.append(_f(h, lab))
    return torch.cat(parts, dim=1)


def policy_gradient_loss_from_token_logprobs(
    token_logprobs: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    valid_token_normalizer: torch.Tensor | float | None = None,
    valid_sample_normalizer: torch.Tensor | float | None = None,
    normalizer: float = 1.0,
    sequence_normalize: bool = False,
    behavior_logprobs: torch.Tensor | None = None,
    loss_fn: str = "cispo",
    cispo_eps: float | None = None,
    clip_eps_low: float | None = None,
    clip_eps_high: float | None = None,
    clip_ratio_c: float | None = None,
    stats: dict | None = None,
    prompt_lengths: torch.Tensor | None = None,
    kl_tau: float = 0.0,
) -> torch.Tensor:
    """Each loss variant builds a per-token OBJECTIVE tensor J (to be maximized); the
    token/sequence/prompt aggregation below is shared and identical across variants.
    CISPO is the only variant whose J is zero at IGNORE positions by construction
    (token_logprobs is masked to 0 there); grpo/gspo objectives are exp(log_ratio)-shaped,
    which is A (not 0) at IGNORE positions (pads, injected interruption markers, where
    behavior holds 0.0 placeholders), so they MUST be explicitly masked."""
    advantage_weight = advantages.to(token_logprobs.dtype).unsqueeze(-1)  # (B, 1)
    kl_per_token = None  # set below when the optional KL anchor is active (behavior-logprob branch only)
    if behavior_logprobs is not None:
        log_ratio = token_logprobs - behavior_logprobs.to(token_logprobs.dtype)
        valid = labels != IGNORE_INDEX
        if loss_fn == "cispo":
            if cispo_eps is None or cispo_eps <= 0:
                raise ValueError("cispo_eps must be a positive float for the cispo loss")
            # CISPO: stop-grad clipped importance weight min(rho, eps_max) multiplies the
            # advantage; only log pi_theta carries the gradient. Clamp in log-space before
            # exp to avoid overflow: exp(min(log_ratio, log eps)) == min(ratio, eps).
            is_weight = torch.exp(torch.clamp(log_ratio, max=math.log(cispo_eps))).detach()
            token_weight = is_weight * advantage_weight  # (B, T)
            per_token_objective = token_logprobs * token_weight
            clip_active = (log_ratio > math.log(cispo_eps)) & valid
        elif loss_fn == "grpo":
            if clip_eps_low is None or clip_eps_high is None:
                raise ValueError("clip_eps_low/clip_eps_high are required for the grpo loss")
            # PPO-clip surrogate (GRPO/DAPO): gradient flows through the ratio itself.
            # min in objective space == max in loss space (verl compute_policy_loss_vanilla).
            ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
            obj_unclipped = ratio * advantage_weight
            obj_clipped = torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high) * advantage_weight
            objective = torch.minimum(obj_unclipped, obj_clipped)
            clip_active = (obj_clipped < obj_unclipped) & valid
            if clip_ratio_c is not None:
                # Dual-clip PPO (arXiv 1912.09729): for A<0 the unclipped term -rho*|A| is unbounded
                # below as rho grows; floor the objective at c*A so one stale/mismatched token
                # can't dominate the batch gradient.
                objective = torch.where(advantage_weight < 0,
                                        torch.maximum(objective, clip_ratio_c * advantage_weight),
                                        objective)
            per_token_objective = objective.masked_fill(~valid, 0.0)
        elif loss_fn == "gspo":
            if clip_eps_low is None or clip_eps_high is None:
                raise ValueError("clip_eps_low/clip_eps_high are required for the gspo loss")
            # GSPO (arXiv 2507.18071): length-normalized SEQUENCE ratio s_i, clipped; the
            # stop-grad identity logpi - sg(logpi) + sg(log s_i) makes the forward value equal
            # s_i at every token while the gradient distributes over the sequence's tokens.
            seq_lengths = valid.sum(dim=-1).clamp(min=1).to(token_logprobs.dtype)
            log_seq_ratio = log_ratio.masked_fill(~valid, 0.0).sum(dim=-1) / seq_lengths
            log_token_seq_ratio = token_logprobs - token_logprobs.detach() + log_seq_ratio.detach().unsqueeze(-1)
            seq_ratio = torch.exp(torch.clamp(log_token_seq_ratio, max=10.0))
            obj_unclipped = seq_ratio * advantage_weight
            obj_clipped = torch.clamp(seq_ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high) * advantage_weight
            per_token_objective = torch.minimum(obj_unclipped, obj_clipped).masked_fill(~valid, 0.0)
            clip_active = (obj_clipped < obj_unclipped) & valid
        else:
            raise ValueError(f"unknown loss_fn: {loss_fn!r}")
        # Optional prime-rl-style mismatch-KL anchor: penalize kl_tau * mean(log_ratio**2) over
        # valid tokens, aggregated per-branch with the SAME denominator as the policy objective (so
        # kl_tau means the same thing across token/sequence/prompt modes). Grad flows through
        # token_logprobs only (behavior_logprobs is detached). Mask to valid tokens because log_ratio
        # is not necessarily 0 at IGNORE positions (token_logprobs is, but behavior may not be).
        if kl_tau != 0.0:
            kl_per_token = (log_ratio * log_ratio).masked_fill(labels == IGNORE_INDEX, 0.0)
        if stats is not None:
            # Off-policy drift diagnostics over valid (generated) tokens. Loss-agnostic except
            # is_clipped_count, which counts the tokens where THIS loss's clip is active.
            with torch.no_grad():
                raw_ratio = torch.exp(torch.clamp(log_ratio, max=20.0))
                # Keep diagnostics on-device inside the microbatch loop. Converting these to
                # Python scalars here forces a GPU sync for every microbatch.
                stats["is_ratio_sum"] = ((raw_ratio * valid).sum()).detach()
                stats["is_clipped_count"] = (clip_active.sum()).detach()
                stats["valid_tokens"] = valid.sum().detach()
                # Train/infer mismatch diagnostics (vLLM bf16 behavior vs fp32 trainer recompute):
                # Σ log_ratio gives the mean log-ratio drift; Σ w² (with Σ w = is_ratio_sum) gives the
                # effective sample size (Σw)²/(N·Σw²). Both on-device, no GPU sync (see comment above).
                stats["log_ratio_sum"] = ((log_ratio * valid).sum()).detach()
                stats["is_ratio_sq_sum"] = (((raw_ratio * raw_ratio) * valid).sum()).detach()
    else:
        token_weight = advantage_weight              # (B, 1) broadcasts
        per_token_objective = token_logprobs * token_weight
    if prompt_lengths is not None:
        # ScaleRL prompt-level aggregation (ScaleRL §3.2, the J_ScaleRL objective): each
        # sample's token-summed loss is divided by its PROMPT GROUP's total valid-token count L_p
        # (prompt_lengths — the same value for every sample of a group), then averaged over the
        # prompts. prompt_lengths is computed over the FULL batch on rank 0 and carried per-sample,
        # so the result is correct even when a group is split across ranks/microbatches; the global
        # prompt count P arrives via valid_sample_normalizer. per_token_objective is 0 at
        # IGNORE positions (see the loss branches above), so the per-sample sum spans valid tokens.
        denom = prompt_lengths.to(token_logprobs.dtype).clamp(min=1.0)
        prompt_normalizer = labels.size(0) if valid_sample_normalizer is None else valid_sample_normalizer
        pg_objective = (per_token_objective.sum(dim=-1) / denom).sum() / (prompt_normalizer * normalizer)
        loss = -pg_objective
        if kl_per_token is not None:
            kl_objective = (kl_per_token.sum(dim=-1) / denom).sum() / (prompt_normalizer * normalizer)
            loss = loss + kl_tau * kl_objective
        return loss
    if sequence_normalize:
        valid_mask = labels != IGNORE_INDEX
        sample_lengths = valid_mask.sum(dim=-1).clamp(min=1).to(token_logprobs.dtype)
        sample_normalizer = labels.size(0) if valid_sample_normalizer is None else valid_sample_normalizer
        pg_objective = (per_token_objective.sum(dim=-1) / sample_lengths).sum() / (sample_normalizer * normalizer)
        loss = -pg_objective
        if kl_per_token is not None:
            kl_objective = (kl_per_token.sum(dim=-1) / sample_lengths).sum() / (sample_normalizer * normalizer)
            loss = loss + kl_tau * kl_objective
        return loss
    valid_count = (labels != IGNORE_INDEX).sum().clamp(min=1)
    token_normalizer = valid_count if valid_token_normalizer is None else valid_token_normalizer
    pg_objective = per_token_objective.sum() / (token_normalizer * normalizer)
    loss = -pg_objective
    if kl_per_token is not None:
        kl_objective = kl_per_token.sum() / (token_normalizer * normalizer)
        loss = loss + kl_tau * kl_objective
    return loss


def chunked_policy_loss(model, input_ids, attention_mask, labels, advantages, *,
                        behavior_logprobs=None, loss_fn="cispo", cispo_eps=None,
                        clip_eps_low=None, clip_eps_high=None, clip_ratio_c=None,
                        chunk_size=1024,
                        valid_token_normalizer=None, valid_sample_normalizer=None,
                        normalizer=1.0, sequence_normalize=False, stats=None,
                        autocast_dtype=None, prompt_lengths=None, kl_tau=0.0):
    """Memory-efficient policy loss (cispo/grpo/gspo): run the base model to hidden states
    (small: B,T,H), then apply the LM head + cross-entropy in checkpointed sequence chunks to
    avoid materializing the full (T, vocab) fp32 logits. The head always runs in fp32; with
    autocast_dtype set, only the body forward is bf16 (halving activation memory). In the
    fp32 path (autocast_dtype=None) this equals computing the full logits in one shot and
    feeding them through policy_gradient_loss_from_token_logprobs."""
    base = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if base is None or lm_head is None:
        raise AttributeError("chunked_policy_loss requires model.model + model.lm_head (HF CausalLM)")
    with torch.autocast(device_type=input_ids.device.type, dtype=autocast_dtype,
                        enabled=autocast_dtype is not None):
        outputs = base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = outputs.last_hidden_state  # (B, T, H); bf16 when autocast_dtype is set
    token_logprobs = chunked_token_logprobs(lm_head, hidden, labels, chunk_size=chunk_size)
    return policy_gradient_loss_from_token_logprobs(
        token_logprobs, labels, advantages,
        valid_token_normalizer=valid_token_normalizer,
        valid_sample_normalizer=valid_sample_normalizer,
        normalizer=normalizer, sequence_normalize=sequence_normalize,
        behavior_logprobs=behavior_logprobs, loss_fn=loss_fn, cispo_eps=cispo_eps,
        clip_eps_low=clip_eps_low, clip_eps_high=clip_eps_high, clip_ratio_c=clip_ratio_c,
        stats=stats, prompt_lengths=prompt_lengths, kl_tau=kl_tau)


def batch_normalize_advantages(advantages: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """ScaleRL batch-level advantage normalization: scale (already per-puzzle centered)
    advantages by the batch std. Safe when std == 0 (returns ~zeros)."""
    std = advantages.std(unbiased=False)
    return advantages / (std + eps)


def group_has_signal(rewards, eps: float = 1e-8) -> bool:
    """A puzzle group yields gradient only if its rewards vary (non-zero advantage)."""
    r = torch.as_tensor(list(rewards) if not isinstance(rewards, torch.Tensor) else rewards,
                        dtype=torch.float32)
    if r.numel() == 0:
        return False
    return bool((r.max() - r.min()).item() > eps)


def is_fresh_enough(example_version: int, current_version: int, max_staleness: int) -> bool:
    """True if a rollout generated at example_version is within max_staleness weight-versions
    of the current trainer version."""
    return (current_version - example_version) <= max_staleness


# ============================ WEIGHT SYNC (trainer rank 0 -> vLLM child) ============================
class PipelineWeightSync:
    """Trainer-side (rank 0) half of vLLM 0.22.0 native NCCL weight transfer."""

    def __init__(self, model, world_size: int):
        from vllm.utils.network_utils import get_ip, get_open_port
        self.model = model
        self.world_size = world_size
        self.master_address = get_ip()
        self.master_port = get_open_port()
        self._group = None
        self._metadata = None

    def init_group(self):
        # Blocks until the child's init_weight_transfer_engine rendezvouses. Trainer == rank 0.
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
        self._group = NCCLWeightTransferEngine.trainer_init(dict(
            master_address=self.master_address,
            master_port=self.master_port,
            world_size=self.world_size,
        ))

    def _named_parameters(self):
        # torch.compile wraps the body and inserts an `_orig_mod.` segment into param names; strip it
        # so the broadcast names match what the vLLM generator expects (no-op when --compile is off).
        for name, p in self.model.named_parameters():
            yield name.replace("_orig_mod.", ""), p

    def metadata(self):
        # Param names/dtypes/shapes are invariant across training steps, so build the
        # lists once and reuse — this runs on the per-step rank-0 weight-sync path.
        if self._metadata is None:
            names, dtype_names, shapes = [], [], []
            for name, p in self._named_parameters():
                names.append(name)
                dtype_names.append(str(p.dtype).split(".")[-1])
                shapes.append(list(p.shape))
            self._metadata = (names, dtype_names, shapes)
        return self._metadata

    def send(self):
        """PRODUCER. Blocking NCCL broadcast; run via asyncio.to_thread / a thread so it
        overlaps the child's update_weights recv. Drives ONLY the PyNccl weight-transfer group —
        the daemon thread that calls this must NEVER issue a dist.* (trainer-group) collective."""
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine, NCCLTrainerSendWeightsArgs,
        )
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=self._named_parameters(),
            trainer_args=NCCLTrainerSendWeightsArgs(group=self._group, packed=True),
        )


# ---------------------------------------------------------------------------
# Held-out evaluation (the leaderboard metric). Decoupled from the training
# signal: a fixed, held-out, reproducible pass@1/pass@k on the eval set, sized
# for a tight CI. See run_standalone_eval / `--eval-only`.
# ---------------------------------------------------------------------------

def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021): probability that at least one of k
    samples drawn without replacement from n total (c correct) is correct."""
    if k > n:
        raise ValueError(f"pass@k requires k<=n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (used for the greedy k==1 case)."""
    if n <= 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _bootstrap_ci(values: list[float], *, n_boot: int = 10000, seed: int = 0,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-puzzle solve fractions. Puzzle-level
    resampling captures both between-puzzle and within-puzzle (sampling) variance."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    try:
        import numpy as _np
        rng = _np.random.default_rng(seed)
        arr = _np.asarray(values, dtype=float)
        means = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
        return (float(_np.quantile(means, alpha / 2)), float(_np.quantile(means, 1.0 - alpha / 2)))
    except Exception:
        rng = random.Random(seed)
        means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
        return (means[int((alpha / 2) * n_boot)], means[min(n_boot - 1, int((1.0 - alpha / 2) * n_boot))])


async def run_held_out_eval(
    engine,
    tokenizer,
    eval_task,
    *,
    k: int,
    sampling: dict,
    enable_thinking: bool = True,
    indices: list[int] | None = None,
    pass_at_ks: tuple[int, ...] = (1, 4, 8, 16),
    concurrency: int = 128,
    progress_every: int = 200,
    collect_rollouts: bool = False,
) -> dict:
    """Evaluate the policy served by `engine` on `eval_task`: k samples per puzzle, each scored
    with reasoning_gym's binary solve check via `eval_task.evaluate`. Returns pass@1, unbiased
    pass@k, per-puzzle solve fractions and a confidence interval. Pure measurement (no logprobs,
    no training); reuses the same prompt/scoring path as the trainer for an apples-to-apples set.
    With collect_rollouts, also returns result["rollouts"]: one row per sample (completion text,
    extracted moves, solved, finish_reason) so records can be re-scored offline (verify_record.py)."""
    if indices is None:
        indices = list(range(len(eval_task)))
    eval_sampling = dict(sampling)
    eval_sampling["logprobs"] = 0  # eval needs no per-token logprobs
    sem = asyncio.Semaphore(concurrency)
    n_total = len(indices)
    done = 0

    async def _one(idx: int) -> dict:
        nonlocal done
        conv = eval_task[idx]
        prompt_ids = encode_prompt(tokenizer, conv["question"], enable_thinking)
        async with sem:
            samples = await generate_group(engine, prompt_ids.tolist(), k, eval_sampling)
        c = answered = extract_fail = length_trunc = 0
        rows: list[dict] = []
        for j, s in enumerate(samples):
            completion = tokenizer.decode(s["token_ids"], skip_special_tokens=True)
            moves = extract_sokoban_answer(completion)
            if moves is None:
                extract_fail += 1
            else:
                answered += 1
            if s.get("finish_reason") == "length":
                length_trunc += 1
            solved = eval_task.evaluate(conv, completion)
            c += solved
            if collect_rollouts:
                rows.append({
                    "puzzle_idx": idx, "sample": j, "solved": solved, "moves": moves,
                    "finish_reason": s.get("finish_reason"),
                    "gen_tokens": len(s["token_ids"]), "completion": completion,
                })
        done += 1
        if progress_every and done % progress_every == 0:
            print(f"  eval progress: {done}/{n_total} puzzles", flush=True)
        return {
            "n": len(samples),
            "c": c,
            "answered": answered,
            "extract_fail": extract_fail,
            "length_trunc": length_trunc,
            "rows": rows,
        }

    results = await asyncio.gather(*[_one(i) for i in indices])

    per_puzzle = [r["c"] / r["n"] if r["n"] else 0.0 for r in results]
    n_puzzles = len(per_puzzle)
    total_samples = sum(r["n"] for r in results)
    total_solved = sum(r["c"] for r in results)
    total_answered = sum(r["answered"] for r in results)
    total_length_trunc = sum(r["length_trunc"] for r in results)
    pass_at_1 = sum(per_puzzle) / max(1, n_puzzles)  # puzzle-weighted (== solved/total when n_i==k)

    # vLLM can drop occasional None/empty completions (see _vllm_generate), so a puzzle may carry
    # n < k samples. Clamp the estimator's k to the per-puzzle n: pass@min(j, n) <= pass@j, so a
    # shortfall can only deflate the score (never inflate a record), and the eval cannot crash at
    # aggregation (the estimator requires k <= n) after all the generation work is done.
    n_short = sum(1 for r in results if r["n"] < k)
    if n_short:
        print(f"WARNING: {n_short}/{n_puzzles} puzzles returned fewer than k={k} samples; "
              f"pass@k clamps k to each puzzle's sample count", flush=True)
    pass_at_k = {
        j: sum(_pass_at_k_unbiased(r["n"], r["c"], min(j, r["n"])) for r in results) / max(1, n_puzzles)
        for j in pass_at_ks if j <= k
    }

    if n_puzzles > 1:
        var = sum((v - pass_at_1) ** 2 for v in per_puzzle) / (n_puzzles - 1)
        se = math.sqrt(var / n_puzzles)
    else:
        se = 0.0

    if k == 1:
        ci_low, ci_high = _wilson_ci(total_solved, total_samples)
    else:
        ci_low, ci_high = _bootstrap_ci(per_puzzle, seed=int(sampling.get("seed") or 0))

    return {
        "n_puzzles": n_puzzles,
        "k": k,
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "per_puzzle_solve_frac": per_puzzle,
        "per_puzzle_n": [r["n"] for r in results],
        "per_puzzle_solved_count": [r["c"] for r in results],
        "per_puzzle_answered_count": [r["answered"] for r in results],
        "per_puzzle_length_trunc_count": [r["length_trunc"] for r in results],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "n_extract_fail": sum(r["extract_fail"] for r in results),
        "n_answered": total_answered,
        "n_length_trunc": total_length_trunc,
        "answer_rate": total_answered / max(1, total_samples),
        "solve_given_answer": total_solved / max(1, total_answered),
        "trunc_frac": total_length_trunc / max(1, total_samples),
        "sampling": eval_sampling,
        **({"rollouts": [row for r in results for row in r["rows"]]} if collect_rollouts else {}),
    }


# --- Student-t survival function P(T >= t) for the record significance test; scipy fast-path,
# --- pure-stdlib fallback (Numerical Recipes incomplete beta). -------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for mloop in range(1, MAXIT + 1):
        m2 = 2 * mloop
        aa = mloop * (b - mloop) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + mloop) * (qab + mloop) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """One-sided upper-tail probability P(T >= t) for a Student-t with `df` degrees of freedom."""
    try:
        from scipy.stats import t as _t  # fast, exact path when available
        return float(_t.sf(t, df))
    except Exception:
        pass
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    ib = _betai(df / 2.0, 0.5, x)  # = I_x(df/2, 1/2)
    return 0.5 * ib if t > 0 else 1.0 - 0.5 * ib


def _eval_one_checkpoint(args: argparse.Namespace, model_path: str, multi: bool) -> dict:
    """Authoritative held-out evaluation of one checkpoint (or the base model), decoupled from
    training: builds its own vLLM engine sized for the full rollout budget (and shuts it down
    afterwards), runs the leaderboard pass@1/pass@k protocol on the fixed eval set, and writes a
    per-checkpoint JSON. Returns the JSON record. No torchrun/DDP."""
    eval_task = load_sokoban_jsonl_dataset(args.eval_data, split_name="eval")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    indices = None
    if args.eval_limit is not None and args.eval_limit > 0:
        indices = list(range(min(args.eval_limit, len(eval_task))))

    sampling = dict(
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        top_k=args.eval_top_k,
        max_tokens=args.eval_max_tokens,
        seed=args.eval_seed,
        logprobs=0,
    )
    if args.eval_interruption:
        marker_ids = tokenizer(INTERRUPTION_TEXT, add_special_tokens=False)["input_ids"]
        sampling["interrupt"] = make_interrupt_config(
            marker_ids, args.eval_interrupt_answer_tokens, args.eval_max_model_len,
            base_temperature=args.eval_temperature, base_top_p=args.eval_top_p,
            base_top_k=args.eval_top_k,
            temperature=args.eval_interrupt_temperature,
            top_p=args.eval_interrupt_top_p, top_k=args.eval_interrupt_top_k,
        )

    print(
        f"[eval] model={model_path} eval_data={args.eval_data} "
        f"n={len(indices) if indices is not None else len(eval_task)} k={args.eval_k} "
        f"max_tokens={args.eval_max_tokens} sampling=temp{args.eval_temperature}/top_p{args.eval_top_p}/"
        f"top_k{args.eval_top_k}/seed{args.eval_seed} "
        f"interruption={args.eval_interruption}",
        flush=True,
    )

    async def _run() -> dict:
        engine = build_async_engine(
            model_path,
            num_dp=args.eval_vllm_dp,
            dtype="bfloat16",
            max_model_len=args.eval_max_model_len,
            gpu_memory_utilization=args.eval_gpu_mem_util,
            seed=args.eval_seed,
            enforce_eager=args.eval_vllm_enforce_eager,
            **vllm_tuning_from_args(args, prefix="eval_"),
        )
        try:
            return await run_held_out_eval(
                engine, tokenizer, eval_task,
                k=args.eval_k, sampling=sampling,
                enable_thinking=args.enable_thinking,
                indices=indices,
                concurrency=args.eval_concurrency,
                collect_rollouts=args.eval_save_rollouts,
            )
        finally:
            try:
                engine.shutdown()
            except Exception:
                pass

    result = asyncio.run(_run())

    step = args.eval_step
    if step is None:
        m = re.search(r"step_?(\d+)", model_path)
        step = int(m.group(1)) if m else None

    record = {
        "seed": args.eval_seed,
        "run": args.run,
        "step": step,
        "checkpoint": model_path,
        "model": args.model,
        "eval_data": str(args.eval_data),
        "git_commit": _git_commit(),
        **{key: result[key] for key in (
            "n_puzzles", "k", "pass_at_1", "pass_at_k", "ci_low", "ci_high", "se",
            "n_extract_fail", "n_answered", "n_length_trunc", "answer_rate",
            "solve_given_answer", "trunc_frac", "sampling", "per_puzzle_solve_frac",
            "per_puzzle_n", "per_puzzle_solved_count", "per_puzzle_answered_count",
            "per_puzzle_length_trunc_count",
        )},
    }

    if args.eval_output is not None:
        out_path = Path(args.eval_output)
    else:
        run_name = args.run if (args.run and args.run != "dummy") else "eval"
        safe = sanitize_run_name(run_name)
        suffix = f"step{step:06d}" if step is not None else "latest"
        if multi:
            # Disambiguate per-checkpoint JSONs by the checkpoint's run dir (outputs/<run>/step_NNNNNN).
            ckpt_tag = sanitize_run_name(Path(model_path).parent.name or model_path)
            out_path = args.output_dir / safe / f"eval_{ckpt_tag}_{suffix}.json"
        else:
            out_path = args.output_dir / safe / f"eval_{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Eval rollouts artifact: full completions for offline re-scoring (verify_record.py). Written
    # BEFORE the JSON so the record can name its artifact; gzipped (~3x) since completions dominate.
    rollouts = result.get("rollouts")
    if rollouts is not None:
        rollouts_path = out_path.with_name(out_path.stem + ".rollouts.jsonl.gz")
        meta = {
            "type": "meta", "seed": args.eval_seed, "run": args.run, "step": step,
            "checkpoint": model_path, "model": args.model,
            "eval_data": str(args.eval_data), "eval_data_sha256": _file_sha256(args.eval_data),
            "k": args.eval_k, "n_puzzles": result["n_puzzles"],
            "git_commit": _git_commit(), "sampling": result["sampling"],
        }
        with gzip.open(rollouts_path, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for row in rollouts:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        record["rollouts_file"] = rollouts_path.name
        print(f"[eval] saved {len(rollouts)} eval rollouts -> {rollouts_path}", flush=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")

    pk = " ".join(f"pass@{j}={result['pass_at_k'][j]:.4f}" for j in sorted(result["pass_at_k"]))
    print(
        f"[eval] {model_path} | n={result['n_puzzles']} k={result['k']} | "
        f"pass@1={result['pass_at_1']:.4f} (95% CI [{result['ci_low']:.4f}, {result['ci_high']:.4f}], "
        f"se={result['se']:.4f}) | {pk} | "
        f"answer_rate={result['answer_rate']:.4f} solve|answer={result['solve_given_answer']:.4f} "
        f"trunc={result['trunc_frac']:.4f} | extract_fail={result['n_extract_fail']} "
        f"length_trunc={result['n_length_trunc']} | -> {out_path}",
        flush=True,
    )
    return record


def run_standalone_eval(args: argparse.Namespace) -> None:
    """Single record-eval entrypoint: evaluates each --eval-checkpoint (one final checkpoint per
    training seed) sequentially under the same protocol, writes one JSON per checkpoint, and —
    given >=2 checkpoints — runs the leaderboard significance test (one-sided t-test that the
    mean pass@1 clears --eval-target) and exits 0/1 on PASS/FAIL."""
    checkpoints = [str(c) for c in args.eval_checkpoint] if args.eval_checkpoint else [args.model]
    if len(set(checkpoints)) < len(checkpoints):
        sys.exit(f"error: duplicate --eval-checkpoint paths {checkpoints} — each checkpoint must "
                 "come from a distinct training run (the training run is the unit of replication).")
    if args.eval_output is not None and len(checkpoints) > 1:
        sys.exit("error: --eval-output is only valid with a single checkpoint; multi-checkpoint "
                 "record evals auto-name one JSON per checkpoint.")

    records = [_eval_one_checkpoint(args, mp, multi=len(checkpoints) > 1) for mp in checkpoints]
    if len(records) < 2:
        return

    values = [float(r["pass_at_1"]) for r in records]
    K = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)  # ddof=1: checkpoint-to-checkpoint (seed-to-seed) variance
    se = sd / math.sqrt(K)
    df = K - 1
    if se == 0.0:
        t = math.inf if mean > args.eval_target else (-math.inf if mean < args.eval_target else 0.0)
        p = 0.0 if mean > args.eval_target else (1.0 if mean < args.eval_target else 0.5)
    else:
        t = (mean - args.eval_target) / se
        p = student_t_sf(t, df)
    passed = p < args.eval_alpha and mean > args.eval_target

    print(f"\n=== Record significance: mean(pass@1) > {args.eval_target} ===", flush=True)
    print(f"  checkpoints (K) : {K}  {[round(v, 4) for v in values]}")
    print(f"  mean +/- sd     : {mean:.4f} +/- {sd:.4f}   (se={se:.4f})")
    print(f"  t / df          : {t:.3f} / {df}")
    print(f"  one-sided p     : {p:.5f}   (alpha={args.eval_alpha})")
    print("  decision        : " + ("PASS  "
          f"(mean {mean:.4f} clears {args.eval_target} with p={p:.5f} < {args.eval_alpha})" if passed else
          f"FAIL  (need mean>{args.eval_target} and p<{args.eval_alpha}; raise mean or add seeds)"),
          flush=True)
    raise SystemExit(0 if passed else 1)


# ============================ MODEL LOAD / SAVE ============================
class FP32LMHead(torch.nn.Module):
    """LM head that holds an fp32 weight and upcasts hidden states on the fly.

    Constructed from an existing `nn.Linear` (typically `model.lm_head` after the
    base model is loaded in bf16). Detaches and clones the weight so it stands
    independent of any tied embedding parameter.
    """

    def __init__(self, lm_head: torch.nn.Linear):
        super().__init__()
        self.in_features = lm_head.in_features
        self.out_features = lm_head.out_features
        self.weight = torch.nn.Parameter(
            lm_head.weight.detach().clone().to(torch.float32)
        )
        if lm_head.bias is not None:
            self.bias = torch.nn.Parameter(
                lm_head.bias.detach().clone().to(torch.float32)
            )
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.to(self.weight.dtype), self.weight, self.bias)


_TORCH_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
        return torch.bfloat16 if bf16_ok else torch.float32
    try:
        return _TORCH_DTYPES[dtype_name]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from None


def _resolve_flash_attention_2_or_raise() -> None:
    """Fail fast (with an actionable message) if attn_implementation='flash_attention_2' cannot be
    satisfied, instead of crashing mid-run on the first forward. transformers 5.8 can serve FA2 from
    the HF `kernels` hub when the standalone flash-attn package is absent (the case here), but that
    needs the `kernels` library + a one-time hub download of kernels-community/flash-attn2."""
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            return  # native flash-attn build present
    except Exception:
        pass
    try:
        from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
    except Exception:
        # Resolver API absent in this transformers version; let from_pretrained resolve at load time.
        return
    try:
        from transformers.utils import is_kernels_available
        kernels_ok = is_kernels_available()
    except Exception:
        kernels_ok = True  # let lazy_import_flash_attention be the judge
    if not kernels_ok:
        raise RuntimeError(
            "--attn-implementation flash_attention_2 needs either the flash-attn package or the HF "
            "`kernels` library (pip install kernels), or use --attn-implementation sdpa."
        )
    try:
        lazy_import_flash_attention("kernels-community/flash-attn2")
    except Exception as exc:
        raise RuntimeError(
            "--attn-implementation flash_attention_2: failed to load the kernels-hub flash-attn2 "
            f"kernel (needs a one-time HF-hub download / network access): {exc!r}. Pre-download it "
            "on the node or use --attn-implementation sdpa."
        ) from exc


def load_model_and_tokenizer(args: argparse.Namespace, device: torch.device):
    dtype = resolve_torch_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    added_pad_token = False
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            added_pad_token = True

    loader_kwargs = dict(trust_remote_code=args.trust_remote_code)
    # When flash_attention_2 (the default) is requested, resolve the kernel up front; if it's
    # unavailable (no `kernels` pkg / no network / unsupported GPU), fall back to SDPA with a
    # warning rather than crash the run — the fallback is visible in the log + as a higher train_s.
    # fp32 head/tie is unaffected: the attention impl only runs the bf16-autocast body matmul.
    requested_attn = args.attn_implementation
    if requested_attn == "flash_attention_2":
        try:
            _resolve_flash_attention_2_or_raise()
        except Exception as exc:
            print0(f"WARNING: --attn-implementation flash_attention_2 unavailable; falling back to sdpa: {exc}")
            requested_attn = "sdpa"
    if requested_attn != "sdpa":
        loader_kwargs["attn_implementation"] = requested_attn
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **loader_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **loader_kwargs)
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print0(f"Attention implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")

    if dtype != torch.float32 and hasattr(model, "lm_head") and isinstance(model.lm_head, torch.nn.Linear):
        model.lm_head = FP32LMHead(model.lm_head)
        model.lm_head.to(device)
        if getattr(model.config, "tie_word_embeddings", False):
            model.config.tie_word_embeddings = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    if args.compile:
        # Compile the transformer BODY only (the fp32 head + chunked CE stay eager — they run under
        # their own checkpoint with autocast disabled). dynamic=True avoids recompiles on the varying
        # per-microbatch sequence lengths (var-prefix batches); watch train_s for recompile churn.
        model.model = torch.compile(model.model, dynamic=True)
        print0("torch.compile enabled on the transformer body (dynamic=True)")

    return model, tokenizer


def strip_compiled_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remove torch.compile's `_orig_mod.` segment from state-dict keys.

    With --compile, model.model is an OptimizedModule, so the raw state dict names body
    keys `model._orig_mod.layers...`. transformers does NOT strip that prefix on load, so
    from_pretrained would silently random-init the entire body (warning only) and any eval
    of the checkpoint would score garbage. Mirrors PipelineWeightSync._named_parameters,
    which is what keeps the live vLLM weight sync compile-safe; no-op when --compile is off.
    """
    return {name.replace("_orig_mod.", ""): value for name, value in state_dict.items()}


def save_hf_checkpoint(
    model,
    tokenizer,
    output_dir: Path,
    step: int,
) -> Path:
    checkpoint_dir = output_dir / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dict = strip_compiled_state_dict_keys(model.state_dict())
    # Belt-and-braces: a compiled-module prefix in the saved keys means from_pretrained
    # would silently drop those weights — refuse to write such a checkpoint.
    leaked = [name for name in state_dict if "_orig_mod" in name]
    if leaked:
        raise RuntimeError(f"compiled-module prefix leaked into checkpoint keys: {leaked[:3]}")
    model.save_pretrained(checkpoint_dir, state_dict=state_dict)
    tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def should_save_checkpoint_for_step(step: int, num_steps: int, save_every: int, save_final: bool) -> bool:
    final_step = step == num_steps - 1
    periodic_save = save_every > 0 and step > 0 and step % save_every == 0
    return periodic_save or (final_step and save_final)


# ============================ RECORD-KEEPING (wandb stub, run log, attestation) ============================
class DummyWandb:
    def log(self, *args, **kwargs):
        pass

    def define_metric(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def sanitize_run_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe or "run"


def _file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


class RunLogger:
    """Self-contained record log, modded-nanogpt style: rank 0 writes outputs/<run>/log_<8hex>.txt
    holding the full script source, environment attestation (versions, git commit, nvidia-smi,
    argv/config), per-step record-clock lines, in-loop eval lines, and a final line stamped when
    the final checkpoint finishes writing — one file proves what ran, on what, with what result.
    The record clock starts at the START of the first training step (startup is untimed; see the
    README rules). Disabled instances (non-rank-0) are inert, and logging can never crash a run:
    every I/O failure prints a warning and permanently disables the logger."""

    _DIVIDER = "=" * 100
    _STOP = object()

    def __init__(self, run_dir: Path, args: argparse.Namespace, enabled: bool) -> None:
        self._fh = None
        self._queue: queue.Queue[str | object] | None = None
        self._thread: threading.Thread | None = None
        self._t0: float | None = None
        if not enabled:
            return
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            self.path = run_dir / f"log_{uuid.uuid4().hex[:8]}.txt"
            self._fh = self.path.open("w", encoding="utf-8")
            self._fh.write(self._header(args))
            self._fh.flush()
            self._queue = queue.Queue()
            self._thread = threading.Thread(target=self._writer_main, name="runlogger-writer", daemon=True)
            self._thread.start()
            print0(f"RunLogger: writing record log to {self.path}")
        except Exception as exc:
            print0(f"RunLogger disabled (failed to open log file): {exc!r}")
            self.close()

    @staticmethod
    def _header(args: argparse.Namespace) -> str:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except Exception as exc:
            source = f"<source unavailable: {exc!r}>"
        env_lines = []
        env_lines.append("python: " + " ".join(sys.version.split()))
        env_lines.append(f"torch: {torch.__version__}")
        for pkg in ("transformers", "vllm"):
            try:
                import importlib.metadata
                env_lines.append(f"{pkg}: {importlib.metadata.version(pkg)}")
            except Exception:
                env_lines.append(f"{pkg}: unknown")
        env_lines.append(f"git commit: {_git_commit()}")
        try:
            import subprocess
            smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
            env_lines.append(smi.stdout if smi.returncode == 0 else f"nvidia-smi failed: {smi.stderr}")
        except Exception as exc:
            env_lines.append(f"nvidia-smi unavailable: {exc!r}")
        env_lines.append(f"argv: {sys.argv}")
        env_lines.append(f"args: {vars(args)}")
        d = RunLogger._DIVIDER
        return f"{source}\n{d}\n" + "\n".join(env_lines) + f"\n{d}\n"

    def _write(self, line: str) -> None:
        q = self._queue
        if q is None:
            return
        try:
            q.put_nowait(line)
        except Exception as exc:
            print0(f"RunLogger disabled (enqueue failed): {exc!r}")
            self.close()

    def _writer_main(self) -> None:
        q = self._queue
        fh = self._fh
        if q is None or fh is None:
            return
        try:
            while True:
                item = q.get()
                try:
                    if item is self._STOP:
                        break
                    fh.write(str(item) + "\n")
                    fh.flush()
                finally:
                    q.task_done()
        except Exception as exc:
            print0(f"RunLogger disabled (write failed): {exc!r}")
        finally:
            try:
                fh.close()
            except Exception:
                pass
            self._fh = None
            if self._thread is threading.current_thread():
                self._queue = None

    def start_clock(self, anchor: float) -> None:
        if self._t0 is None:
            self._t0 = anchor

    def record_time(self) -> float:
        return time.monotonic() - self._t0 if self._t0 is not None else 0.0

    def log_step(self, step: int, num_steps: int, metrics: dict) -> None:
        rt = self.record_time()
        self._write(
            f"step:{step + 1}/{num_steps} record_time:{rt:.1f}s step_avg:{rt / (step + 1):.1f}s "
            f"reward_mean:{metrics.get('reward/mean', float('nan')):.4f} "
            f"solved_frac:{metrics.get('reward/solved_frac', float('nan')):.4f} "
            f"loss:{metrics.get('loss', float('nan')):.4f} "
            f"grad_norm:{metrics.get('grad_norm', float('nan')):.4f}"
        )

    def log_eval(self, round_step: int, metrics: dict) -> None:
        self._write(
            f"eval step:{round_step} "
            f"pass@1:{metrics.get('eval/pass_at_1', float('nan')):.4f} "
            f"answer_rate:{metrics.get('eval/answer_rate', float('nan')):.4f} "
            f"solve_given_answer:{metrics.get('eval/solve_given_answer', float('nan')):.4f} "
            f"trunc_frac:{metrics.get('eval/trunc_frac', float('nan')):.4f} "
            f"record_time:{self.record_time():.1f}s"
        )

    def log_final_checkpoint(self, checkpoint_dir: Path) -> None:
        self._write(f"final_checkpoint:{checkpoint_dir} record_time:{self.record_time():.1f}s")

    def close(self) -> None:
        q = self._queue
        thread = self._thread
        self._queue = None
        self._thread = None
        if q is not None:
            try:
                q.put_nowait(self._STOP)
            except Exception:
                pass
        if thread is not None:
            thread.join(timeout=30)
            if thread.is_alive():
                print0("RunLogger close timed out; background writer still draining")
        elif self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None


# ============================ TRAINING PIPELINE ============================
def _safe_qsize(q) -> float:
    """qsize() is unimplemented on some platforms; report -1 instead of crashing."""
    try:
        return float(q.qsize())
    except (NotImplementedError, OSError):
        return -1.0


def ewma_update(prev: float, value: float, alpha: float, *, seed: bool) -> float:
    """One EWMA step; `seed` (e.g. during warmup) resets the average to the raw value."""
    return value if seed else (1.0 - alpha) * prev + alpha * value


@dataclass
class CollectStats:
    """Rank-0 Phase-A bookkeeping for one step's collect loop (run_pipeline's
    _collect_step_batch). The *_total/_unfiltered counters are the unbiased online proxies,
    counted over ALL fresh generated groups BEFORE the zero-variance filter; the lists are
    aligned per accepted (trained) sample."""

    step_examples: list[RolloutExample] = field(default_factory=list)
    finish_reasons: list[str | None] = field(default_factory=list)
    staleness: list[int] = field(default_factory=list)
    group_solved: list[float] = field(default_factory=list)  # 1.0 if any sample in the group solved
    groups_seen_total: int = 0
    samples_seen_total: int = 0
    samples_answered_total: int = 0
    samples_solved_total: int = 0
    samples_length_trunc_unfiltered: int = 0
    samples_interrupted_total: int = 0           # forced-answer marker survived into the scored sequence
    samples_interrupted_answered_total: int = 0  # ... and the sample still produced a parseable answer
    groups_any_solved_total: int = 0
    gen_tokens_total: int = 0
    prefill_tokens_total: int = 0
    puzzles_used: int = 0
    groups_zero_variance: int = 0
    groups_zv_allfail: int = 0
    groups_zv_allpass: int = 0
    groups_stale: int = 0
    consecutive_rejected: int = 0
    consecutive_rejected_max: int = 0

    def register_reject(self, step: int, reject_limit: int) -> None:
        """One rejected group (stale / zero-variance / unusable): bump the consecutive-reject
        counter and hard-abort past reject_limit (degenerate-generation safety net)."""
        self.consecutive_rejected += 1
        self.consecutive_rejected_max = max(self.consecutive_rejected_max, self.consecutive_rejected)
        if self.consecutive_rejected > reject_limit:
            raise RuntimeError(
                f"step {step}: rejected {self.consecutive_rejected} consecutive rollout groups "
                f"(staleness bound too tight or rewards degenerate); aborting")


class InloopEval:
    """Rank-0 in-loop held-out eval, run through the SHARED training vLLM engine.

    Eval prompts are enqueued with puzzle ids >= EVAL_PID_BASE so the Phase-A collect loop can
    route their results here instead of training on them. Per-round per-puzzle counts accumulate
    as results stream back (interleaved with training rollouts, possibly across several train
    steps); a round is logged ONLY once every puzzle returned — a partial round is never logged
    as eval/pass_at_1, only warned about past --inloop-eval-deadline-steps."""

    def __init__(self, args: argparse.Namespace, tokenizer, eval_task: FixedSokobanDataset,
                 interrupt_marker_ids: list[int] | None, prompt_q, wandb_run, run_logger) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.eval_task = eval_task
        self.interrupt_marker_ids = interrupt_marker_ids
        self.prompt_q = prompt_q
        self.wandb_run = wandb_run
        self.run_logger = run_logger
        self.prompts: list[dict[str, Any]] = []
        self.rounds: dict[int, dict[str, Any]] = {}
        self.pid_to_round: dict[int, tuple[int, int]] = {}
        self.next_eval_pid = EVAL_PID_BASE

    def prepare_prompts(self) -> None:
        """Pre-tokenize the eval set once (rank-0 setup) and validate it fits max_model_len."""
        args = self.args
        max_prompt_len = 0
        for idx in range(len(self.eval_task)):
            conv = self.eval_task[idx]
            prompt_ids = encode_prompt(self.tokenizer, conv["question"], args.enable_thinking)
            max_prompt_len = max(max_prompt_len, int(prompt_ids.numel()))
            self.prompts.append({
                "index": idx,
                "conversation": conv,
                "prompt_ids": prompt_ids.tolist(),
            })
        if max_prompt_len + args.inloop_eval_max_tokens > args.max_model_len:
            raise ValueError(
                f"--max-model-len {args.max_model_len} too small for in-loop eval: "
                f"max eval prompt ({max_prompt_len}) + --inloop-eval-max-tokens "
                f"({args.inloop_eval_max_tokens}) exceeds it"
            )
        if not isinstance(self.wandb_run, DummyWandb):
            self.wandb_run.define_metric("eval/*", step_metric="eval/round_step")
        print0(
            f"In-loop held-out eval ON: every={args.inloop_eval_every} steps, "
            f"n={len(self.prompts)}, k={args.inloop_eval_k}, "
            f"max_tokens={args.inloop_eval_max_tokens}, "
            f"interruption={args.inloop_eval_interruption}"
        )

    def _sampling(self) -> dict[str, Any]:
        args = self.args
        sampling_eval = {
            "temperature": args.eval_temperature,
            "top_p": args.eval_top_p,
            "top_k": args.eval_top_k,
            "max_tokens": args.inloop_eval_max_tokens,
            "seed": args.inloop_eval_seed,
            "logprobs": 0,
        }
        if args.inloop_eval_interruption:
            if self.interrupt_marker_ids is None:
                raise RuntimeError("--inloop-eval-interruption requested but interruption marker was not initialized")
            sampling_eval["interrupt"] = make_interrupt_config(
                self.interrupt_marker_ids, args.inloop_eval_interrupt_answer_tokens, args.max_model_len,
                base_temperature=args.eval_temperature, base_top_p=args.eval_top_p,
                base_top_k=args.eval_top_k,
                temperature=args.eval_interrupt_temperature,
                top_p=args.eval_interrupt_top_p, top_k=args.eval_interrupt_top_k,
            )
        return sampling_eval

    def launch_round(self, round_step: int, weight_version: int) -> None:
        args = self.args
        if args.inloop_eval_every <= 0 or not self.prompts:
            return
        if round_step in self.rounds:
            return
        state = {
            "round_step": round_step,
            "launched_at_step": round_step,
            "weight_version": weight_version,
            "expected": len(self.prompts),
            "received": 0,
            "n_samples": 0,
            "n_answered": 0,
            "n_solved": 0,
            "n_length_trunc": 0,
            "pids": [],
            "per_puzzle_n": [0] * len(self.prompts),
            "per_puzzle_solved_count": [0] * len(self.prompts),
            "per_puzzle_answered_count": [0] * len(self.prompts),
            "per_puzzle_length_trunc_count": [0] * len(self.prompts),
        }
        self.rounds[round_step] = state
        sampling_eval = self._sampling()
        for idx, item in enumerate(self.prompts):
            pid = self.next_eval_pid
            self.next_eval_pid += 1
            state["pids"].append(pid)
            self.pid_to_round[pid] = (round_step, idx)
            self.prompt_q.put({
                "type": MSG_GENERATE,
                "puzzle_id": pid,
                "prompt_token_ids": item["prompt_ids"],
                "num_samples": args.inloop_eval_k,
                "sampling": sampling_eval,
            })
        print0(
            f"launched in-loop eval for step {round_step}: "
            f"{len(self.prompts)} puzzles x k={args.inloop_eval_k}"
        )

    def process_result(self, msg: dict[str, Any], current_step: int) -> None:
        pid = int(msg["puzzle_id"])
        route = self.pid_to_round.pop(pid, None)
        if route is None:
            return
        round_step, eval_idx = route
        state = self.rounds.get(round_step)
        if state is None:
            return

        conv = self.prompts[eval_idx]["conversation"]
        n = solved = answered = length_trunc = 0
        for sample in msg["samples"]:
            n += 1
            completion = self.tokenizer.decode(sample["token_ids"], skip_special_tokens=True)
            if extract_sokoban_answer(completion) is not None:
                answered += 1
            if sample.get("finish_reason") == "length":
                length_trunc += 1
            solved += self.eval_task.evaluate(conv, completion)

        state["received"] += 1
        state["n_samples"] += n
        state["n_answered"] += answered
        state["n_solved"] += solved
        state["n_length_trunc"] += length_trunc
        state["per_puzzle_n"][eval_idx] = n
        state["per_puzzle_solved_count"][eval_idx] = solved
        state["per_puzzle_answered_count"][eval_idx] = answered
        state["per_puzzle_length_trunc_count"][eval_idx] = length_trunc

        if state["received"] >= state["expected"]:
            self._finalize_round(round_step, current_step)

    def warn_overdue(self, current_step: int) -> None:
        """Flag (once per round) rounds still incomplete past the deadline; never logs them."""
        for round_step, state in self.rounds.items():
            if current_step - int(state["launched_at_step"]) < self.args.inloop_eval_deadline_steps:
                continue
            if state.get("deadline_warned"):
                continue
            state["deadline_warned"] = True
            print0(
                f"in-loop eval step {round_step:3d} still pending after "
                f"{self.args.inloop_eval_deadline_steps} train steps "
                f"({int(state['received'])}/{int(state['expected'])} puzzles); "
                "not logging eval/pass_at_1 until the full eval set completes"
            )

    def _finalize_round(self, round_step: int, current_step: int) -> None:
        state = self.rounds.pop(round_step, None)
        if state is None:
            return
        for pid in state["pids"]:
            self.pid_to_round.pop(pid, None)

        received = int(state["received"])
        expected = int(state["expected"])
        if received != expected:
            raise RuntimeError(
                f"refusing to finalize incomplete in-loop eval round {round_step}: "
                f"received {received}/{expected} puzzles"
            )
        per_puzzle = [
            c / n
            for c, n in zip(state["per_puzzle_solved_count"], state["per_puzzle_n"])
            if n > 0
        ]
        total_samples = int(state["n_samples"])
        total_answered = int(state["n_answered"])
        total_solved = int(state["n_solved"])
        total_trunc = int(state["n_length_trunc"])
        pass_at_1 = sum(per_puzzle) / max(1, len(per_puzzle))
        answer_rate = total_answered / max(1, total_samples)
        solve_given_answer = total_solved / max(1, total_answered)
        trunc_frac = total_trunc / max(1, total_samples)
        metrics = {
            "eval/round_step": round_step,
            "eval/logged_step": current_step,
            "eval/round_weight_version": state["weight_version"],
            "eval/round_complete_frac": received / max(1, expected),
            "eval/received_puzzles": received,
            "eval/expected_puzzles": expected,
            "eval/pass_at_1": pass_at_1,
            "eval/answer_rate": answer_rate,
            "eval/solve_given_answer": solve_given_answer,
            "eval/trunc_frac": trunc_frac,
            "eval/n_samples": total_samples,
            "eval/n_answered": total_answered,
            "eval/n_solved": total_solved,
            "eval/n_length_trunc": total_trunc,
            "eval/finalize_reason_complete": 1.0,
        }
        self.wandb_run.log(metrics)
        self.run_logger.log_eval(round_step, metrics)
        print0(
            f"in-loop eval step {round_step:3d} (complete, {received}/{expected} puzzles) | "
            f"pass@1 {pass_at_1:.3f} answer {answer_rate:.3f} "
            f"solve|answer {solve_given_answer:.3f} trunc {trunc_frac:.3f}"
        )


def run_pipeline(
    args: argparse.Namespace,
    train_task: FixedSokobanDataset | None = None,
    eval_task: FixedSokobanDataset | None = None,
) -> None:
    import multiprocessing as mp

    if train_task is None:
        train_task = load_sokoban_jsonl_dataset(args.train_data, split_name="train")
    if eval_task is None:
        eval_task = load_sokoban_jsonl_dataset(args.eval_data, split_name="eval")

    if args.dtype != "float32":
        # Non-fp32 triggers the FP32LMHead wrapper, which unties lm_head from embed_tokens on
        # tied-embedding models (e.g. all Qwen3). The trainer would then learn a separate
        # lm_head.weight that the still-tied vLLM generator skips on weight sync, desyncing the
        # generator head from the trained policy. fp32 keeps the tie (only embed_tokens.weight
        # syncs) and yields fp32 logits inherently.
        raise ValueError(
            "training requires --dtype float32 (non-fp32 untie of lm_head desyncs the "
            "vLLM generator head from the trained policy on tied-embedding models)"
        )

    # Trainer ranks select their GPU via set_device(LOCAL_RANK) over all visible devices; a
    # preset CUDA_VISIBLE_DEVICES would break that physical rank->GPU mapping. The vLLM child
    # gets its own CVD from the layout below only for the child process start.
    assert "CUDA_VISIBLE_DEVICES" not in os.environ, (
        "run with CUDA_VISIBLE_DEVICES unset so each trainer rank maps to physical GPU "
        "LOCAL_RANK and the vLLM child can claim GPUs T..T+M-1"
    )
    # The recipe fills one NODE_GPUS-GPU node: torchrun WORLD_SIZE is the trainer count T
    # (1 when launched without torchrun), and the vLLM generators take the rest, M = NODE_GPUS - T.
    # That makes `torchrun --nproc_per_node=T -m speedrun` the whole launch, no GPU-split flags.
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    # world_size == T trainer ranks. Rank 0 (master_process) is the only rank that owns the vLLM
    # child / weight-sync / queues / wandb; ranks 1..T-1 are pure compute workers that only join
    # the per-step collectives (scatter -> all_reduce counts -> all_reduce grads -> barrier).
    # Every collective below is guarded by `if world_size > 1`, so T==1 (no torchrun) runs the
    # single-GPU path with no process group initialized — byte-identical to before the DP port.
    world_size = ddp_world_size
    master_process = ddp_rank == 0
    vllm_dp = NODE_GPUS - world_size
    if vllm_dp < 1:
        raise ValueError(
            f"need at least 1 GPU for vLLM generators; {world_size} trainer ranks fill the "
            f"whole {NODE_GPUS}-GPU node (use --nproc_per_node < {NODE_GPUS})"
        )

    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)          # pin trainer to physical GPU LOCAL_RANK before spawning the child

    # TWO-NCCL-GROUP RENDEZVOUS ORDER (load-bearing). The trainer runs two disjoint NCCL groups:
    # (a) the TRAINER group (torch.distributed default over ranks 0..T-1, for grad all-reduce +
    # per-step scatter/barrier), and (b) the WEIGHT-TRANSFER group (vLLM's PipelineWeightSync
    # PyNccl group, rank 0 <-> vLLM child, on its own address/port). They share no communicator,
    # but the bring-up order must be pinned so no worker sits in a trainer-group collective while
    # rank 0 is blocked rendezvousing group (b) with the child. Required order:
    #   (1) dist.init_process_group + dist.barrier()                  -- THIS block, all ranks
    #   (2) rank 0: spawn child -> _await_status(MSG_ENGINE_READY)
    #   (3) rank 0: wsync.init_group()  (blocks on the child's PyNccl rendezvous)
    #   (4) rank 0: _await_status(MSG_ENGINE_READY)  (weight-transfer engine ready)
    #   (5) all ranks: rank-0 init-status broadcast after setup succeeds or fails
    # NO trainer-group collective may occur between (2) and (4) — the workers are past the (1)
    # barrier and only wait at the (5) broadcast, which rank 0 sources after finishing (4). The
    # whole block is skipped for T==1.
    if world_size > 1:
        pg_timeout = timedelta(seconds=TRAINER_PG_TIMEOUT_S)
        try:
            dist.init_process_group(backend="nccl", device_id=device, timeout=pg_timeout)
        except TypeError:
            dist.init_process_group(backend="nccl", timeout=pg_timeout)
        dist.barrier()  # (1) the ONLY init barrier; symmetric all-rank; see invariant above.
    print0(f"Pipeline trainer world size: {world_size}")

    set_seed(args.seed, device)
    torch.set_float32_matmul_precision("high")
    # The cuDNN SDPA/MHA graph path has failed intermittently in multi-trainer backward
    # (`mha_graph.execute(...).is_good()`), which then deadlocks the manual grad all-reduce.
    # Prefer Flash/mem-efficient/math SDPA backends for trainer reliability.
    if device.type == "cuda" and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    model, tokenizer = load_model_and_tokenizer(args, device)
    train_autocast_dtype = (
        torch.bfloat16 if args.train_autocast_dtype == "bfloat16" and device.type == "cuda" else None
    )
    model.config.use_cache = False
    model.train()
    pad_token_id = tokenizer.pad_token_id
    if len(train_task) < 1:
        raise ValueError("No training examples available")
    if len(eval_task) < 1:
        raise ValueError("No eval examples available")
    print0(f"Loaded datasets: train={len(train_task)} from {train_task.path}; eval={len(eval_task)} from {eval_task.path}")

    optimizer = build_optimizer(model.parameters(), args)
    steps_per_epoch = max(1, len(train_task) // args.examples_per_step)
    num_steps = steps_per_epoch * args.num_epochs
    if args.max_steps is not None:
        num_steps = min(num_steps, args.max_steps)
    if num_steps < 1:
        raise ValueError("No training steps requested")
    run_dir = args.output_dir / sanitize_run_name(args.run)
    for group in optimizer.param_groups:
        group["initial_lr"] = args.learning_rate

    # RANK-0-ONLY state: the vLLM child, PipelineWeightSync, the queues, `current_version`, the
    # puzzle bookkeeping + enqueue closure, wandb and the rollout file all live on rank 0. Workers
    # never touch them; they only join the per-step collectives (Phase C/D/E/G).
    child = wsync = None
    prompt_q = result_q = control_q = status_q = None
    current_version = 0
    puzzles: dict[int, dict] = {}
    next_puzzle_id = 0
    # FIFO reorder state: results arrive in completion order, but training consumes groups in
    # strict dataset-file order (the published ordering IS the curriculum; completion-order
    # consumption biased early steps toward fast-finishing easy groups — the cold-start reward
    # dip artifact — and stale-dropped the slowest ~5% of groups). pid -> (result msg, puzzle
    # meta). Bounded by --inflight-requests: the replacement ticket is issued at consume time.
    reorder_buffer: dict[int, tuple[dict, dict]] = {}
    next_consume_pid = 0
    train_prompt_cache: list[PromptCacheEntry] = []
    interrupt_marker_ids: list[int] | None = None
    if args.interruption or args.inloop_eval_interruption:
        interrupt_marker_ids = tokenizer(INTERRUPTION_TEXT, add_special_tokens=False)["input_ids"]
    sampling = dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    max_tokens=args.max_new_tokens)
    if args.interruption:
        assert interrupt_marker_ids is not None
        sampling["interrupt"] = make_interrupt_config(
            interrupt_marker_ids, args.interrupt_answer_tokens, args.max_model_len,
            base_temperature=args.temperature, base_top_p=args.top_p, base_top_k=args.top_k,
            temperature=args.interrupt_temperature,
            top_p=args.interrupt_top_p, top_k=args.interrupt_top_k,
            phase1_min_tokens=args.interrupt_min_tokens,
            phase1_max_tokens=args.interrupt_max_tokens,
        )
        # Guard: the phase-2 prompt is prompt + phase-1 cap + marker + answer; leave headroom for the
        # prompt so it fits max_model_len (the per-request clamp in generate_group is the runtime backstop).
        phase1_max_tokens = (
            args.interrupt_max_tokens
            if args.interrupt_max_tokens is not None
            else args.max_new_tokens
        )
        phase1_min_tokens = (
            args.interrupt_min_tokens
            if args.interrupt_min_tokens is not None
            else phase1_max_tokens
        )
        need = phase1_max_tokens + len(interrupt_marker_ids) + args.interrupt_answer_tokens
        if args.max_model_len < need + 256:
            raise ValueError(
                f"--max-model-len {args.max_model_len} too small for --interruption: needs >= prompt + "
                f"phase-1 cap({phase1_max_tokens}) + marker({len(interrupt_marker_ids)}) + "
                f"interrupt-answer-tokens({args.interrupt_answer_tokens}); raise --max-model-len.")
        if args.interrupt_min_tokens is not None or args.interrupt_max_tokens is not None:
            phase1_desc = (
                f"random[{phase1_min_tokens}, {phase1_max_tokens}]"
                if phase1_min_tokens < phase1_max_tokens
                else str(phase1_max_tokens)
            )
        else:
            phase1_desc = str(args.max_new_tokens)
        print0(
            f"Interruption-based length control ON: marker={len(interrupt_marker_ids)} tok, "
            f"phase-1 budget={phase1_desc}, answer budget={args.interrupt_answer_tokens}"
        )
    wandb_run = DummyWandb()
    model_perf_info = None
    run_logger = RunLogger(run_dir=run_dir, args=args, enabled=master_process)
    inloop_eval: InloopEval | None = None  # rank 0 only; built in _rank0_setup (needs queues + wandb)

    def enqueue_puzzle(pid):
        if not train_prompt_cache:
            raise RuntimeError("train prompt cache is not initialized")
        cached = train_prompt_cache[pid % len(train_prompt_cache)]
        puzzles[pid] = {
            "conversation": cached.conversation,
            "prompt_ids": cached.prompt_ids,
            "prefix_length": cached.prefix_length,
        }
        prompt_q.put({"type": MSG_GENERATE, "puzzle_id": pid,
                      "prompt_token_ids": cached.prompt_token_ids, "num_samples": args.num_samples,
                      "sampling": sampling})

    def _enqueue_next_puzzle() -> None:
        nonlocal next_puzzle_id
        enqueue_puzzle(next_puzzle_id)
        next_puzzle_id += 1

    def _rank0_setup() -> None:
        nonlocal train_prompt_cache
        nonlocal child, wsync, prompt_q, result_q, control_q, status_q
        nonlocal wandb_run, model_perf_info, rollout_stream_q, rollout_stream_thread, inloop_eval
        # Rendezvous steps (2)-(4); see the two-NCCL-group invariant above. All of this runs on
        # rank 0 only, with NO dist.* collective between (2) and (4): the workers are parked at the
        # (5) init-status broadcast, so a stray trainer-group collective here would have no peer
        # and hang the run.
        #
        # --- (2) spawn the vLLM child on GPUs T..T+M-1 ---
        layout = pipeline_trainer_layout(ddp_world_size, vllm_dp)
        train_prompt_cache = build_prompt_cache(train_task, tokenizer, args.enable_thinking)
        max_train_prompt_len = max(entry.prefix_length for entry in train_prompt_cache)
        print0(
            f"Precomputed train prompt cache: {len(train_prompt_cache)} prompts "
            f"(max {max_train_prompt_len} tokens)"
        )
        ctx = mp.get_context("spawn")
        prompt_q, result_q, control_q, status_q = ctx.Queue(), ctx.Queue(), ctx.Queue(maxsize=0), ctx.Queue()
        visible = layout.visible_gpus
        cfg = dict(
            model=args.model, num_dp=vllm_dp,
            dtype="bfloat16",  # vLLM generates in bf16; the trainer may be fp32 (CISPO + FP32 logits head absorb the mismatch)
            max_model_len=args.max_model_len, gpu_memory_utilization=args.vllm_gpu_mem_util,
            seed=args.seed, inflight=args.inflight_requests, visible_gpus=visible,
            enforce_eager=args.vllm_enforce_eager,
            **vllm_tuning_from_args(args),
        )
        child = ctx.Process(target=engine_child_main, args=(cfg, prompt_q, result_q, control_q, status_q))
        _start_process_with_cuda_visible_devices(child, visible)
        _await_status(status_q, MSG_ENGINE_READY)        # (2) engine constructed (pre weight-group init)

        # --- (3)/(4) build the weight-sync group (trainer rank 0 <-> child workers ranks 1..K) ---
        # No trainer-group (dist.*) collective between here and the step loop's first Phase-C scatter.
        wsync = PipelineWeightSync(model, world_size=vllm_dp + 1)
        control_q.put({"type": MSG_INIT_WEIGHTS, "master_address": wsync.master_address,
                       "master_port": wsync.master_port, "world_size": vllm_dp + 1})
        wsync.init_group()                                  # (3) blocks on the child's PyNccl rendezvous
        _await_status(status_q, MSG_ENGINE_READY)        # (4) weight-transfer engine ready

        for _ in range(args.inflight_requests):
            _enqueue_next_puzzle()

        # wandb / rollout files / model-perf are RANK-0-ONLY (only rank 0 logs metrics and
        # owns the collect loop). This is part of the guarded rank-0 init path so workers learn
        # about W&B or filesystem setup failures before waiting for the first step scatter.
        if args.run != "dummy":
            import wandb
            wandb_run = wandb.init(project="nanochat-rl-hf", name=args.run, config=vars(args))
        model_perf_info = collect_model_perf_info(model)
        if args.save_rollouts:
            run_dir.mkdir(parents=True, exist_ok=True)
            rollout_stream_q = queue.Queue()
            rollout_stream_thread = threading.Thread(
                target=_rollout_stream_writer, name="rollout-stream", daemon=True)
            rollout_stream_thread.start()
            print0(f"Streaming rollouts to {run_dir / 'rollouts.jsonl.gz'} "
                   f"(async, every {args.rollout_flush_every} steps)")
        inloop_eval = InloopEval(args, tokenizer, eval_task, interrupt_marker_ids,
                                 prompt_q, wandb_run, run_logger)
        if args.inloop_eval_every > 0:
            inloop_eval.prepare_prompts()
    wandb_rollouts_enabled = False
    wandb_rollout_rows: list[list] = []
    wandb_rollout_columns = ["step", "weight_version", "staleness", "puzzle_id", "status",
                             "reward", "advantage", "gen_tokens", "finish_reason", "completion"]
    # Last-N-steps FULL rollouts (bounded record artifact, written once at teardown; crash
    # coverage comes from the rollout stream below, which holds the same rows and more).
    final_rollout_buf: deque[list[dict]] | None = (
        deque(maxlen=args.final_rollout_steps) if master_process and args.final_rollout_steps > 0 else None
    )
    # Append-only rollout stream (--save-rollouts): EVERY consumed group (trained + zero-variance)
    # lands in <run_dir>/rollouts.jsonl.gz, batched per --rollout-flush-every steps and appended by
    # a daemon thread — the training loop never does I/O. Each flush is its own gzip member
    # (gzip.open reads concatenated members natively); a kill mid-write costs at most the
    # unflushed tail, so a prematurely stopped run keeps its full history for forensics.
    rollout_stream_pending: list[dict] = []
    rollout_stream_q: queue.Queue | None = None
    rollout_stream_thread: threading.Thread | None = None

    def _rollout_stream_writer() -> None:
        path = run_dir / "rollouts.jsonl.gz"
        while True:
            chunk = rollout_stream_q.get()
            if chunk is None:
                return
            try:
                with gzip.open(path, "at", encoding="utf-8") as fh:
                    for row in chunk:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                print0(f"rollout-stream append failed (non-fatal): {exc!r}")

    def _stream_flush_pending() -> None:
        nonlocal rollout_stream_pending
        if rollout_stream_q is not None and rollout_stream_pending:
            rollout_stream_q.put(rollout_stream_pending)
            rollout_stream_pending = []

    # Abort protocol. `_abort_step` turns a controlled rank-0 reject into a RuntimeError without
    # broadcasting; the single poison broadcast on the failure path comes from `_broadcast_poison`,
    # called from the blanket handler around rank 0's Phase A/B and before each post-G weight-sync
    # raise. `_broadcast_poison` is idempotent within a step (the `poison_sent` holder) and a no-op
    # at world_size==1.
    def _abort_step(message: str) -> "RuntimeError":
        """Return the error for a controlled rank-0 reject; the blanket handler broadcasts."""
        return RuntimeError(message)

    def _broadcast_poison(poison_sent: list[bool]) -> None:
        """Broadcast the poison sentinel to the workers so they unblock at the (next)
        Phase-C broadcast and raise in lockstep. No-op at world_size==1 and idempotent
        within a step (guarded by the single-element `poison_sent` flag holder)."""
        if world_size > 1 and not poison_sent[0]:
            dist.broadcast_object_list(_abort_payload(), src=0)
            poison_sent[0] = True

    rank0_init_status_sent = False

    def _send_rank0_init_status(ok: bool, error: str | None = None) -> None:
        nonlocal rank0_init_status_sent
        if world_size > 1:
            dist.broadcast_object_list(_pipeline_init_status_payload(ok, error), src=0)
        rank0_init_status_sent = True

    def _recv_rank0_init_status() -> None:
        payload = _pipeline_init_status_payload(False, "rank 0 exited before sending init status")
        dist.broadcast_object_list(payload, src=0)
        _raise_for_pipeline_init_status(payload[0])

    # Single worker that double-buffers Phase E's microbatch host tensors (every rank tensorizes
    # its own shard). One worker is deliberate: prefetch depth 1 is enough to hide the CPU build
    # behind fwd/bwd, and the thread must never issue dist.* collectives (tensorize_microbatch_cpu
    # is pure host-side work).
    tensorize_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tensorize")
    # Rank-0-only pool that fans process_rollout_sample out across a group's samples: the trim
    # binary search re-decodes multi-thousand-token prefixes ~12x per answered sample, and fast-
    # tokenizer decode releases the GIL; scoring is thread-safe (reasoning_gym builds a fresh Game
    # per score_answer call — verified no shared mutable state). map() preserves sample order, and
    # exceptions surface on iteration exactly like the old serial loop (same Phase-A blanket handler).
    decode_pool = (
        ThreadPoolExecutor(max_workers=max(1, min(args.num_samples, 8)), thread_name_prefix="decode")
        if master_process else None
    )

    def _collect_step_batch(step: int, t_step_start: float, times: dict[str, float]) -> CollectStats:
        """Phase A (RANK 0 ONLY): bank examples_per_step fresh, signal-bearing rollout groups,
        consuming groups in STRICT DATASET-FILE ORDER via the reorder buffer (results arrive in
        completion order). In-loop-eval results (pid >= EVAL_PID_BASE) are routed out of band;
        stale / zero-variance / unusable groups are rejected in file order (with the reject-limit
        safety abort in CollectStats.register_reject). Raises on engine errors — the caller's
        blanket handler broadcasts the poison sentinel."""
        nonlocal next_consume_pid
        cs = CollectStats()
        reject_limit = max(256, args.examples_per_step * 64)

        def _absorb_result(msg: dict) -> None:
            """Route one result_q message: engine errors raise, in-loop-eval results go out of
            band, train groups land in the reorder buffer keyed by pid. No ticket reissue here —
            that happens at consume time, otherwise the generator forms a closed enqueue loop
            with itself and races arbitrarily far ahead of the trainer."""
            if msg["type"] == MSG_ERROR:
                raise _abort_step(f"engine child error: {msg['msg']}")
            pid = msg["puzzle_id"]
            if pid >= EVAL_PID_BASE:
                inloop_eval.process_result(msg, step)
                return
            reorder_buffer[pid] = (msg, puzzles.pop(pid))

        while cs.puzzles_used < args.examples_per_step:
            head_wait_start = time.monotonic()
            while next_consume_pid not in reorder_buffer:
                if time.monotonic() - head_wait_start > FIFO_HEAD_STALL_ABORT_S:
                    raise _abort_step(
                        f"file-order head group pid={next_consume_pid} not produced after "
                        f"{FIFO_HEAD_STALL_ABORT_S:.0f}s (generator dropped the request?)")
                t_wait_start = time.monotonic()
                try:
                    msg = result_q.get(timeout=ROLLOUT_GET_TIMEOUT_S)
                except queue.Empty:
                    if not child.is_alive():
                        raise _abort_step("vLLM child died during generation (no rollouts received)")
                    continue
                finally:
                    times["result_queue_wait"] += time.monotonic() - t_wait_start
                _absorb_result(msg)
            # Opportunistic drain: absorb everything already finished so result_q stays empty
            # and the buffer (not the IPC queue) holds the inventory.
            while True:
                try:
                    msg = result_q.get_nowait()
                except queue.Empty:
                    break
                _absorb_result(msg)
            msg, meta = reorder_buffer.pop(next_consume_pid)
            next_consume_pid += 1
            _enqueue_next_puzzle()
            pid, version = msg["puzzle_id"], msg["weight_version"]
            if not is_fresh_enough(version, current_version, args.max_staleness):
                cs.groups_stale += 1
                cs.register_reject(step, reject_limit)
                continue
            t_process_start = time.monotonic()
            processed_samples = list(decode_pool.map(
                lambda sample: process_rollout_sample(
                    tokenizer,
                    meta["prompt_ids"],
                    meta["prefix_length"],
                    sample,
                    meta["conversation"],
                    trim_after_answer=args.trim_after_answer,
                ),
                msg["samples"],
            ))
            times["decode_trim_score"] += time.monotonic() - t_process_start
            if not processed_samples:
                # vLLM dropped every completion in this group (see _vllm_generate's
                # None/empty filter): nothing to score, and the zero-variance branch
                # below would crash on max([]). Treat it as a rejected group so the
                # reject-limit safety net still catches a persistently broken generator.
                print0(f"[collect] step {step}: group pid={pid} returned 0 usable samples; skipping")
                cs.register_reject(step, reject_limit)
                continue
            rewards = [sample.reward for sample in processed_samples]
            fins = [sample.finish_reason for sample in processed_samples]
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            adv = rewards_t - rewards_t.mean()
            adv_values = adv.tolist()
            # Unbiased online proxy (before any filter): solved == reward 1.0, matching evaluate().
            cs.groups_seen_total += 1
            cs.samples_seen_total += len(rewards)
            cs.samples_answered_total += sum(
                1 for sample in processed_samples if sample.moves is not None
            )
            cs.samples_solved_total += sum(1 for r in rewards if r == 1.0)
            cs.samples_length_trunc_unfiltered += sum(1 for fin in fins if fin == "length")
            cs.groups_any_solved_total += 1 if any(r == 1.0 for r in rewards) else 0
            # Interrupted = the forced-answer marker survived into the scored sequence (loss_mask
            # carries False on injected positions; a pre-marker self-answer trims the marker away).
            interrupted = [s.loss_mask is not None and not all(s.loss_mask) for s in processed_samples]
            cs.samples_interrupted_total += sum(interrupted)
            cs.samples_interrupted_answered_total += sum(
                1 for s, was in zip(processed_samples, interrupted) if was and s.moves is not None)
            is_zero_var = args.zero_variance_filter and not group_has_signal(rewards)
            t_log_start = time.monotonic()
            if rollout_stream_q is not None:
                # Dict building only — the gz append happens on the stream writer thread. The
                # question text is omitted (puzzle_id + the pinned dataset reproduce it).
                status = "zero_variance" if is_zero_var else "trained"
                for sample, a in zip(processed_samples, adv_values):
                    rollout_stream_pending.append({
                        "step": step, "weight_version": version,
                        "staleness": current_version - version,
                        "puzzle_id": pid, "status": status,
                        "reward": float(sample.reward), "advantage": float(a),
                        "gen_tokens": len(sample.behavior_logprobs),
                        "finish_reason": sample.finish_reason,
                        "completion": sample.completion,
                    })
            times["rollout_logging"] += time.monotonic() - t_log_start
            if is_zero_var:
                cs.groups_zero_variance += 1
                if max(rewards) > 0.5:
                    cs.groups_zv_allpass += 1
                else:
                    cs.groups_zv_allfail += 1
                cs.register_reject(step, reject_limit)
                continue
            cs.consecutive_rejected = 0
            cs.group_solved.append(1.0 if max(rewards) > 0.5 else 0.0)
            cs.prefill_tokens_total += meta["prefix_length"]
            for sample, a in zip(processed_samples, adv_values):
                cs.step_examples.append(RolloutExample(
                    sequence=sample.sequence, prefix_length=meta["prefix_length"],
                    reward=sample.reward, advantage=a,
                    behavior_logprobs=sample.behavior_logprobs,
                    weight_version=version, puzzle_id=pid,
                    completion=sample.completion,
                    loss_mask=sample.loss_mask))
                cs.finish_reasons.append(sample.finish_reason)
                cs.staleness.append(current_version - version)
                cs.gen_tokens_total += len(sample.behavior_logprobs)
            cs.puzzles_used += 1
        times["collect"] = time.monotonic() - t_step_start
        return cs

    def _sync_weights_to_generators(times: dict[str, float]) -> None:
        """RANK 0: push fresh weights to the vLLM generators (MSG_SYNC_WEIGHTS -> threaded
        wsync.send() -> MSG_WEIGHTS_READY) and bump current_version."""
        nonlocal current_version
        t_sync_start = time.monotonic()
        names, dtype_names, shapes = wsync.metadata()
        current_version += 1
        control_q.put({"type": MSG_SYNC_WEIGHTS, "names": names, "dtype_names": dtype_names,
                       "shapes": shapes, "version": current_version})
        send_result: dict = {}

        def _do_send():
            # Daemon-thread constraint: drive ONLY the PyNccl weight-transfer group
            # (wsync.send). Never call dist.* here — the trainer NCCL group is single-
            # threaded on rank 0's main thread and a dist.* call would deadlock it.
            try:
                wsync.send()
            except BaseException as exc:  # surface NCCL/other errors to the parent
                send_result["error"] = exc

        send_thread = threading.Thread(target=_do_send, daemon=True)
        send_thread.start()
        t_send_wait_start = time.monotonic()
        send_thread.join(timeout=WEIGHT_SYNC_TIMEOUT_S)
        times["weight_sync_send"] = time.monotonic() - t_send_wait_start
        if send_thread.is_alive() or not child.is_alive():
            raise RuntimeError(
                "weight broadcast stalled or the vLLM child died during weight sync "
                f"(child_alive={child.is_alive()})"
            )
        if "error" in send_result:
            raise send_result["error"]
        t_status_wait_start = time.monotonic()
        _await_status(status_q, MSG_WEIGHTS_READY)
        times["weight_sync_wait"] = time.monotonic() - t_status_wait_start
        times["sync"] = time.monotonic() - t_sync_start

    def _finish_step_rank0(step: int, lr: float, cs: CollectStats, times: dict[str, float],
                           adv_all: torch.Tensor, loss_sum: float, grad_norm: float,
                           is_ratio_acc: float, is_clip_acc: float, is_tok_acc: int,
                           log_ratio_acc: float, is_ratio_sq_acc: float,
                           global_token_count: int, num_microbatches: int) -> None:
        """RANK 0 end-of-step bookkeeping: metric assembly, wandb/console/record-log lines, the
        EWMA canaries, the live rollout table, checkpoint saving, and the warn/abort guards.
        Raises (e.g. the non-finite grad_norm abort) propagate to the caller's poison handler."""
        nonlocal trunc_ewma, gnorm_ewma, solve_given_answer_ewma, wandb_rollout_rows
        rewards_all = torch.tensor([e.reward for e in cs.step_examples], dtype=torch.float32)
        adv_raw = torch.tensor([e.advantage for e in cs.step_examples], dtype=torch.float32)
        gen_lens = torch.tensor([float(len(e.behavior_logprobs)) for e in cs.step_examples])
        n_length_trunc = sum(1 for f in cs.finish_reasons if f == "length")
        n_eos = sum(1 for f in cs.finish_reasons if f == "stop")
        n_seqs = len(cs.step_examples)
        answer_rate = cs.samples_answered_total / max(1, cs.samples_seen_total)
        solve_given_answer = cs.samples_solved_total / max(1, cs.samples_answered_total)
        if cs.samples_answered_total > 0:
            solve_given_answer_ewma = ewma_update(
                solve_given_answer_ewma if solve_given_answer_ewma is not None else 0.0,
                solve_given_answer, 0.1, seed=solve_given_answer_ewma is None)
        metrics = {
            "step": step, "lr": lr, "weight_version": current_version,
            "groups/used": cs.puzzles_used,
            "groups/zero_variance_dropped": cs.groups_zero_variance,
            "groups/zero_variance_allfail": cs.groups_zv_allfail,
            "groups/zero_variance_allpass": cs.groups_zv_allpass,
            "groups/zero_variance_allfail_frac": cs.groups_zv_allfail / max(1, cs.groups_seen_total),
            "groups/zero_variance_allpass_frac": cs.groups_zv_allpass / max(1, cs.groups_seen_total),
            "groups/stale_dropped": cs.groups_stale,
            # Signal yield this step: accepted groups / all fresh groups seen. Falls as the
            # train set is mastered (all-pass drops climb) or staleness tightens; the
            # high-water mark foreshadows the reject_limit hard abort.
            "groups/acceptance_ratio": cs.puzzles_used / max(1, cs.groups_seen_total),
            "groups/consecutive_rejected_max": cs.consecutive_rejected_max,
            "seqs": n_seqs,
            "reward/mean": float(rewards_all.mean()),
            "reward/std": float(rewards_all.std(unbiased=False)),
            "reward/solved_frac": float((rewards_all > 0.5).float().mean()),
            "reward/group_pass_at_k": float(sum(cs.group_solved) / max(1, len(cs.group_solved))),
            # Unbiased proxies over all fresh generated groups (pre-filter). Use these,
            # not solved_frac/group_pass_at_k, to gauge live progress; see run_held_out_eval.
            "reward/online_solved_frac_unfiltered": cs.samples_solved_total / max(1, cs.samples_seen_total),
            "reward/online_group_any_solved_unfiltered": cs.groups_any_solved_total / max(1, cs.groups_seen_total),
            "reward/answer_rate": answer_rate,
            "reward/solve_given_answer": solve_given_answer,
            "reward/solve_given_answer_ewma": (
                solve_given_answer_ewma if solve_given_answer_ewma is not None else 0.0
            ),
            "adv/raw_std": float(adv_raw.std(unbiased=False)),
            "adv/norm_std": float(adv_all.std(unbiased=False)),
            "loss": loss_sum,
            "grad_norm": grad_norm,
            "train/logits_chunk_size": args.logits_chunk_size,
            "train/device_batch_size": args.device_batch_size,
            "is_ratio/mean": (is_ratio_acc / is_tok_acc) if is_tok_acc else 0.0,
            "is_ratio/clipped_frac": (is_clip_acc / is_tok_acc) if is_tok_acc else 0.0,
            # Effective sample size fraction (Σw)²/(N·Σw²) over valid tokens: 1.0 == on
            # policy, →0 as the bf16-behavior/fp32-trainer mismatch + staleness widen.
            "is_ratio/ess_frac": (
                (is_ratio_acc * is_ratio_acc) / (is_tok_acc * is_ratio_sq_acc)
                if is_tok_acc and is_ratio_sq_acc > 0.0 else 0.0
            ),
            # Mean log(pi_theta / pi_behavior) over valid tokens (signed drift; ~0 = matched).
            "mismatch/mean_log_ratio": (log_ratio_acc / is_tok_acc) if is_tok_acc else 0.0,
            "gen/mean_tokens": float(gen_lens.mean()),
            "gen/max_tokens": float(gen_lens.max()),
            "gen/total_tokens": int(cs.gen_tokens_total),
            "gen/length_trunc_frac": n_length_trunc / max(1, n_seqs),
            "gen/length_trunc_frac_unfiltered": (
                cs.samples_length_trunc_unfiltered / max(1, cs.samples_seen_total)
            ),
            "gen/eos_frac": n_eos / max(1, n_seqs),
            "gen/interrupted_frac": cs.samples_interrupted_total / max(1, cs.samples_seen_total),
            # Of marker-interrupted samples, how many produced a parseable answer. The open-
            # <answer>-tag marker should pin this near 1.0; a sag means the model rambles inside
            # the tag or never closes it. Sparse: omitted on steps with no interruptions.
            **({"reward/answered_given_interrupted":
                cs.samples_interrupted_answered_total / cs.samples_interrupted_total}
               if cs.samples_interrupted_total else {}),
            "staleness/mean": sum(cs.staleness) / max(1, len(cs.staleness)),
            "staleness/max": (max(cs.staleness) if cs.staleness else 0),
            **{f"time/{name}_s": value for name, value in times.items()},
            "throughput/gen_tokens_per_s": cs.gen_tokens_total / max(1e-6, times["collect"]),
            "throughput/train_tokens_per_s": global_token_count / max(1e-6, times["train"]),
            "throughput/seqs_per_s": n_seqs / max(1e-6, times["step"]),
            "throughput/result_q_backlog": _safe_qsize(result_q),
            # Finished groups waiting for their file-order turn; bounded by inflight_requests.
            "throughput/reorder_buffer": len(reorder_buffer),
        }
        metrics.update(estimate_step_perf_metrics(
            model_info=model_perf_info,
            train_padded_tokens=float(global_token_count),
            train_forward_backward_passes=float(num_microbatches),
            rollout_prefill_tokens=float(cs.prefill_tokens_total),
            rollout_decode_tokens=float(cs.gen_tokens_total),
            rollout_forward_passes=float(cs.gen_tokens_total),
            train_seconds=times["train"],
            rollout_seconds=times["collect"],
        ))

        # Truncation EWMA (ScaleRL A.15 leading-spiral indicator): logged to W&B and
        # WARN-only (no hard-abort). During warmup the EWMA just tracks the raw (high,
        # expected) cold-start truncation and is re-seeded at the first post-warmup step,
        # so that history doesn't skew the steady-state value.
        trunc_ewma = ewma_update(trunc_ewma, metrics["gen/length_trunc_frac"],
                                 TRUNC_EWMA_ALPHA, seed=step <= args.warmup_steps)
        metrics["gen/length_trunc_ewma"] = trunc_ewma

        # Grad-norm EWMA: logged + WARN-only (a known collapse mode announces itself as the
        # grad norm climbing ~10x over tens of steps). Mirror the truncation EWMA: seed on
        # warmup, smooth after. A non-finite grad_norm is immediate divergence and still
        # hard-aborts (NaN poisons the weights).
        gnorm_val = float(grad_norm)
        gnorm_finite = math.isfinite(gnorm_val)
        if not gnorm_finite:
            gnorm_ewma = float("inf")
        else:
            gnorm_ewma = ewma_update(gnorm_ewma, gnorm_val, GNORM_EWMA_ALPHA,
                                     seed=step <= args.warmup_steps)
        metrics["grad_norm_ewma"] = gnorm_ewma

        print0(
            f"step {step:3d} | rew {metrics['reward/mean']:.3f} pass@k {metrics['reward/group_pass_at_k']:.2f} "
            f"solved {metrics['reward/solved_frac']:.2f} ans {metrics['reward/answer_rate']:.2f} "
            f"s|a {metrics['reward/solve_given_answer']:.2f} | loss {loss_sum:.4f} gnorm {grad_norm:.2f} "
            f"| IS {metrics['is_ratio/mean']:.2f} clip {metrics['is_ratio/clipped_frac']:.2f} "
            f"| genlen {metrics['gen/mean_tokens']:.0f} trunc {metrics['gen/length_trunc_frac']:.2f} "
            f"| gen {metrics['throughput/gen_tokens_per_s']:.0f} tok/s tr {metrics['throughput/train_tokens_per_s']:.0f} tok/s "
            f"| t {times['step']:.1f}s (gen {times['collect']:.1f}/tr {times['train']:.1f}/sync {times['sync']:.1f}) "
            f"| v{current_version} stale {metrics['staleness/mean']:.1f}"
        )
        if wandb_rollouts_enabled and n_seqs > 0:
            import wandb
            # Sample a spread of this step's rollouts by reward (solved + middle + unsolved).
            examples = cs.step_examples
            order = sorted(range(n_seqs), key=lambda i: examples[i].reward)
            n = min(args.wandb_rollout_samples, n_seqs)
            picks = [order[round(j * (n_seqs - 1) / (n - 1))] for j in range(n)] if n > 1 else [order[0]]
            step_rows = [[step, examples[i].weight_version, cs.staleness[i],
                          examples[i].puzzle_id, "trained", float(examples[i].reward),
                          float(examples[i].advantage), len(examples[i].behavior_logprobs),
                          cs.finish_reasons[i], (examples[i].completion or "")] for i in picks]
            # LIVE per-step rollouts: fold an n-row table into the SAME metrics log (no extra log()
            # call => no wandb-step desync; survives an early kill, unlike the once-at-end dump).
            # FULL completions logged (already decoded; ~few hundred KB/step at N=10, async upload)
            # so the tail — where late repetition / non-termination shows — is visible.
            metrics["rollouts_live"] = wandb.Table(columns=wandb_rollout_columns, data=step_rows)
            wandb_rollout_rows.extend(step_rows)        # running record; full table dumped in finally
            # Cap the cumulative table at 150 steps' worth of the CONFIGURED sample count (the old
            # fixed 1500 assumed N=10 and silently halved coverage at N=20).
            wandb_rollout_rows = wandb_rollout_rows[-150 * max(1, args.wandb_rollout_samples):]
        if final_rollout_buf is not None and n_seqs > 0:
            final_rollout_buf.append([
                {"step": step, "weight_version": e.weight_version, "staleness": cs.staleness[i],
                 "puzzle_id": e.puzzle_id, "reward": float(e.reward), "advantage": float(e.advantage),
                 "gen_tokens": len(e.behavior_logprobs), "finish_reason": cs.finish_reasons[i],
                 "completion": e.completion or ""}
                for i, e in enumerate(cs.step_examples)])
        if args.rollout_flush_every > 0 and step % args.rollout_flush_every == 0:
            _stream_flush_pending()
        wandb_run.log(metrics)
        run_logger.log_step(step, num_steps, metrics)
        inloop_eval.warn_overdue(step)
        if should_save_checkpoint_for_step(step, num_steps, args.save_every, args.save_final):
            checkpoint_dir = save_hf_checkpoint(model, tokenizer, run_dir, step)
            print0(f"Saved checkpoint to {checkpoint_dir}")
            if step == num_steps - 1 and args.save_final:
                # Official record clock stops here: the final checkpoint has finished writing.
                run_logger.log_final_checkpoint(checkpoint_dir)
        if trunc_ewma >= TRUNC_WARN_EWMA:
            print0(f"WARNING step {step}: gen/length_trunc_ewma={trunc_ewma:.3f} "
                   f">= {TRUNC_WARN_EWMA:.2f} (truncation climbing; ScaleRL A.15 spiral risk)")
        if not gnorm_finite:
            raise _abort_step(
                f"non-finite grad_norm ({grad_norm}) at step {step} — divergence; "
                f"aborting; resume from the last saved checkpoint"
            )
        if gnorm_ewma >= GNORM_WARN_EWMA:
            print0(f"WARNING step {step}: grad_norm_ewma={gnorm_ewma:.2f} "
                   f">= {GNORM_WARN_EWMA:.1f} (gradient climbing; explosion risk)")
        if metrics["groups/zero_variance_allpass_frac"] >= ALLPASS_WARN_FRAC:
            print0(f"WARNING step {step}: groups/zero_variance_allpass_frac="
                   f"{metrics['groups/zero_variance_allpass_frac']:.2f} >= {ALLPASS_WARN_FRAC:.2f} "
                   f"(train set being mastered; curriculum starvation / wasted generation)")

    try:
        # RANK-0 INIT STATUS (all ranks, world_size>1 only): workers must not enter the
        # loop's first Phase-C scatter until rank 0 has finished the slow rank-0-only setup.
        # A broadcast, rather than a bare barrier, lets rank 0 report setup failures so workers
        # raise immediately instead of waiting for the NCCL watchdog.
        if master_process:
            try:
                _rank0_setup()
                _send_rank0_init_status(True)
            except BaseException as exc:
                if not rank0_init_status_sent:
                    try:
                        _send_rank0_init_status(False, f"{type(exc).__name__}: {exc}")
                    except BaseException as broadcast_exc:
                        print0(f"failed to broadcast rank-0 pipeline init failure: {broadcast_exc!r}")
                raise
        else:
            _recv_rank0_init_status()
        wandb_rollouts_enabled = args.wandb_rollout_samples > 0 and not isinstance(wandb_run, DummyWandb)

        # Smoothed canaries for the spiral/explosion guards (see TRUNC_*/GNORM_*; updated each step on rank 0).
        trunc_ewma = 0.0
        gnorm_ewma = 0.0
        solve_given_answer_ewma: float | None = None

        for step in range(num_steps):
            has_next_step = step < num_steps - 1
            t_step_start = time.monotonic()
            if step == 0:
                # The record clock starts at the first training step; startup is untimed (see README rules).
                run_logger.start_clock(t_step_start)
            lr = args.learning_rate * get_lr_multiplier(step, num_steps, args.init_lr_frac,
                                                        args.warmup_steps, args.lr_schedule,
                                                        args.min_lr_frac, args.lr_decay_steps,
                                                        args.lr_decay_start)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # PER-STEP COLLECTIVE ORDER. Every trainer rank runs the trainer-group collectives in
            # exactly this order:
            #   A collect (rank 0 only) -> C scatter -> D all_reduce counts -> E backward +
            #   all_reduce grads -> F optimizer.step -> G barrier (strictly before the weight-sync
            #   round-trip) -> [rank 0] MSG_SYNC_WEIGHTS -> wsync.send() -> MSG_WEIGHTS_READY
            # Workers skip Phase A and block at the Phase-C receive. Any rank-0 Phase-A/B failure
            # is caught by the blanket handler, which broadcasts one poison sentinel before
            # re-raising so workers unblock at Phase C and raise in lockstep (no NCCL-watchdog
            # hang). No trainer-group collective may appear between MSG_SYNC_WEIGHTS and
            # MSG_WEIGHTS_READY. Every collective is a guarded no-op at T==1.

            # `poison_sent` guards the Phase-A/B handler and post-G weight-sync path against
            # double-broadcasting the sentinel within a step.
            poison_sent = [False]

            # Per-step wall-clock breakdown, logged as time/<name>_s (initialized in metric
            # order; the collect-side entries are filled inside _collect_step_batch).
            times = {name: 0.0 for name in (
                "collect", "result_queue_wait", "decode_trim_score", "rollout_logging",
                "object_scatter_broadcast", "batch_tensorize", "grad_allreduce",
                "optimizer_clip", "weight_sync_send", "weight_sync_wait",
                "train", "sync", "step",
            )}
            cs: CollectStats | None = None  # rank-0 Phase-A result
            adv_all = None
            shards: list[list[RolloutExample]] | None = None
            adv_shards: list[list[float]] | None = None
            num_prompts_local = 0  # global trained-prompt count P (rank 0 fills it; broadcast in Phase C)
            if master_process:
                # Blanket handler: any exception in rank 0's Phase A/B — explicit `_abort_step`
                # rejects and implicit ones (puzzles.pop KeyError, torch.cat / scoring / trim_*
                # errors, the empty-batch assert) — broadcasts one poison sentinel before
                # propagating, so workers unblock at Phase C and raise in lockstep.
                try:
                    # ---- Phase A (RANK 0 ONLY): collect one step's worth of fresh, signal-
                    # bearing rollouts (see _collect_step_batch above). ----
                    cs = _collect_step_batch(step, t_step_start, times)

                    # The collect loop only exits once `examples_per_step` signal groups landed, so
                    # step_examples is non-empty by construction; guard it before scatter so an
                    # empty batch never reaches sort_and_shard_for_microbatching -> build_rl_batch_varprefix_cpu.
                    assert cs.step_examples, "Phase A produced an empty step batch (would break scatter)"

                    # ---- Phase B (RANK 0 ONLY): batch-normalize advantages over the FULL step batch,
                    # BEFORE scattering. Per-shard std would be wrong, so this must precede the split.
                    adv_all = batch_normalize_advantages(
                        torch.tensor([e.advantage for e in cs.step_examples], dtype=torch.float32, device=device))
                    adv_all_list = adv_all.detach().cpu().tolist()
                    training_step_examples = strip_rollout_completions(cs.step_examples)
                    # ScaleRL prompt-level loss aggregation: tag every sample with its prompt group's
                    # total valid (trainable) token count L_p, summed over the FULL batch HERE — before
                    # the length-sort can split a group across ranks/microbatches — and carried on the
                    # example so it survives the scatter. P = number of trained prompt groups (puzzle
                    # ids). Mirrors local_and_global_counts' valid-token definition (loss_mask sum, or
                    # full generated length when interruption is off). No-op for token/sequence modes.
                    group_token_totals: Counter[int] = Counter()
                    for e in training_step_examples:
                        n_valid = sum(e.loss_mask) if e.loss_mask is not None else len(e.behavior_logprobs)
                        group_token_totals[e.puzzle_id] += n_valid
                    for e in training_step_examples:
                        e.group_token_count = group_token_totals[e.puzzle_id]
                    num_prompts_local = len(group_token_totals)
                    # Length-sort + snake-deal the (example, normalized-advantage) PAIRS across
                    # ranks before the scatter: adjacent microbatch slices get near-equal lengths
                    # (minimal padding) and every rank the same length profile (no straggler at the
                    # Phase-E/F collectives). Examples and advantages travel jointly so their
                    # positional alignment — which the train loop relies on — holds by construction.
                    # Gradient-identical up to fp order (see sort_and_shard_for_microbatching).
                    shards, adv_shards = sort_and_shard_for_microbatching(
                        training_step_examples, adv_all_list, world_size,
                        make_pad_example(pad_token_id))
                except BaseException:
                    # Broadcast the poison sentinel (exactly once) so workers blocked at Phase C
                    # unblock and raise in lockstep, then re-raise the original error on rank 0.
                    _broadcast_poison(poison_sent)
                    raise

            # ---- Phase C (ALL RANKS): scatter rank 0's padded shards (first trainer-group
            # collective of the step): broadcast the full padded batch + advantages, each rank
            # slices its own shard. Element 0 carries the abort flag. ----
            if world_size > 1:
                full_padded = [e for sh in shards for e in sh] if master_process else None
                full_adv = [a for sh in adv_shards for a in sh] if master_process else None
                payload = _scatter_payload(full_padded, full_adv, abort=False,
                                           num_prompts=num_prompts_local)
                t_object_start = time.monotonic()
                dist.broadcast_object_list(payload, src=0)
                times["object_scatter_broadcast"] = time.monotonic() - t_object_start
                control, full_padded, full_adv = payload
                if control["abort"]:
                    raise RuntimeError("aborting after rank-0 poison sentinel (Phase A error on rank 0)")
                global_num_prompts = int(control.get("num_prompts", 0))
                per_rank = len(full_padded) // world_size
                my_shard = full_padded[ddp_rank * per_rank:(ddp_rank + 1) * per_rank]
                my_adv = torch.tensor(
                    full_adv[ddp_rank * per_rank:(ddp_rank + 1) * per_rank],
                    dtype=torch.float32, device=device)
            else:
                # adv_shards[0], NOT adv_all: the shard is length-sorted, and adv_all is still in
                # collection order — indexing it positionally against the sorted shard would
                # silently train on mis-assigned advantages.
                my_shard = shards[0]
                my_adv = torch.tensor(adv_shards[0], dtype=torch.float32, device=device)
                global_num_prompts = num_prompts_local

            # ---- CISPO training step ----
            t_train_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            total = len(my_shard)

            # ---- Phase D (ALL RANKS): per-shard counts -> all_reduce(SUM) -> GLOBAL totals. The
            # loss normalizer needs the PAD-EXCLUDED full-batch counts (SUM over ranks of
            # local_and_global_counts), NOT len(my_shard), to match the single-GPU full-batch
            # gradient. They exclude pads and any zero-token example (which contributes 0 to both
            # numerator and denominator); for real rollouts they equal the single-GPU
            # len(step_examples). Placed after scatter (shard sizes known) and before backward. ----
            local_sample_count, local_token_count = local_and_global_counts(my_shard)
            if world_size > 1:
                counts = torch.tensor([local_sample_count, local_token_count],
                                      dtype=torch.long, device=device)
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                global_sample_count = int(counts[0].item())
                global_token_count = int(counts[1].item())
            else:
                global_sample_count = local_sample_count
                global_token_count = local_token_count
            loss_sum_t = torch.zeros((), dtype=torch.float32, device=device)
            is_ratio_acc_t = torch.zeros((), dtype=torch.float32, device=device)
            is_clip_acc_t = torch.zeros((), dtype=torch.float32, device=device)
            is_tok_acc_t = torch.zeros((), dtype=torch.long, device=device)
            # Mismatch diagnostics (see the stats block in policy_gradient_loss_from_token_logprobs):
            # Σ log_ratio for mean drift, Σ w² for ESS. All-reduced with the IS accumulators below.
            log_ratio_acc_t = torch.zeros((), dtype=torch.float32, device=device)
            is_ratio_sq_acc_t = torch.zeros((), dtype=torch.float32, device=device)
            num_microbatches = 0
            # ---- Phase E (ALL RANKS): backward over THIS rank's shard with normalizer=1.0 and
            # the GLOBAL counts (identical loss-call args to the single-GPU path). The microbatch
            # tensors are double-buffered: a single prefetch thread builds microbatch i+1's host
            # tensors (pinned) while microbatch i runs fwd/bwd, and the main thread does one
            # non_blocking H2D copy per tensor — identical values, no per-row pageable copies.
            # The prefetch thread makes no CUDA-kernel or dist.* calls (see tensorize_microbatch_cpu);
            # all H2D copies stay on the main thread's default stream, preserving kernel order.
            pin_host = device.type == "cuda"
            microbatches = [my_shard[s:s + args.device_batch_size]
                            for s in range(0, total, args.device_batch_size)]
            pending_host = (
                tensorize_pool.submit(tensorize_microbatch_cpu, microbatches[0], pad_token_id, pin_host)
                if microbatches else None
            )
            for mb_index, mb in enumerate(microbatches):
                start = mb_index * args.device_batch_size
                t_tensorize_start = time.monotonic()
                ids_host, attn_host, labels_host, beh_host = pending_host.result()
                if mb_index + 1 < len(microbatches):
                    pending_host = tensorize_pool.submit(
                        tensorize_microbatch_cpu, microbatches[mb_index + 1], pad_token_id, pin_host)
                # Pinned + non_blocking: torch's caching host allocator event-tracks the source
                # buffers, so they are not recycled before the async copy completes.
                input_ids = ids_host.to(device, non_blocking=True)
                attention_mask = attn_host.to(device, non_blocking=True)
                labels = labels_host.to(device, non_blocking=True)
                beh = beh_host.to(device, non_blocking=True)
                times["batch_tensorize"] += time.monotonic() - t_tensorize_start
                mb_stats: dict = {}
                prompt_mode = args.loss_normalization == "prompt"
                # ScaleRL prompt-level: per-sample group-total token count L_p (carried on each
                # example), divided in-loss; the global prompt count P is the normalizer. Pads
                # have group_token_count 0 -> clamp(min=1) in-loss with a 0 numerator => no effect.
                prompt_lengths = (
                    torch.tensor([e.group_token_count for e in mb], dtype=torch.float32, device=device)
                    if prompt_mode else None
                )
                loss = chunked_policy_loss(
                    model, input_ids, attention_mask, labels, my_adv[start:start + len(mb)],
                    behavior_logprobs=beh, loss_fn=args.loss_fn, cispo_eps=args.cispo_eps,
                    clip_eps_low=args.clip_eps_low, clip_eps_high=args.clip_eps_high,
                    clip_ratio_c=args.clip_ratio_c, chunk_size=args.logits_chunk_size,
                    valid_token_normalizer=global_token_count,
                    valid_sample_normalizer=(global_num_prompts if prompt_mode else global_sample_count),
                    normalizer=1.0,
                    sequence_normalize=(args.loss_normalization == "sequence"),
                    stats=mb_stats,
                    autocast_dtype=train_autocast_dtype,
                    prompt_lengths=prompt_lengths,
                    kl_tau=args.kl_tau,
                )
                loss.backward()
                loss_sum_t = loss_sum_t + loss.detach()
                tok = mb_stats.get("valid_tokens")
                if tok is not None:
                    is_ratio_acc_t = is_ratio_acc_t + mb_stats["is_ratio_sum"].to(torch.float32)
                    is_clip_acc_t = is_clip_acc_t + mb_stats["is_clipped_count"].to(torch.float32)
                    is_tok_acc_t = is_tok_acc_t + tok.to(torch.long)
                    log_ratio_acc_t = log_ratio_acc_t + mb_stats["log_ratio_sum"].to(torch.float32)
                    is_ratio_sq_acc_t = is_ratio_sq_acc_t + mb_stats["is_ratio_sq_sum"].to(torch.float32)
                num_microbatches += 1
            # ---- Phase E (end): manual DP grad reduction — all_reduce(SUM) every param grad so
            # each rank holds the full-batch gradient Σ_r grad(per-shard loss), exactly the
            # single-GPU gradient given normalizer=1.0 + global counts.
            if world_size > 1:
                t_allreduce_start = time.monotonic()
                all_reduce_grads_sum(model)
                times["grad_allreduce"] = time.monotonic() - t_allreduce_start
                # Reduce the diagnostic accumulators to GLOBAL sums in the all-ranks region (one
                # extra small all-reduce). Without this they were rank-0-local, so is_ratio/* only
                # reflected rank 0's shard; the new ESS/drift metrics need the full-batch view too.
                # loss_sum_t rides along: each rank computes its shard loss with normalizer=1.0
                # over GLOBAL counts, so the true step loss is the SUM over ranks — without this
                # the logged `loss` (wandb + record log) was rank 0's shard-partial value (~1/T).
                diag_t = torch.stack([
                    is_ratio_acc_t, is_clip_acc_t, is_tok_acc_t.to(torch.float32),
                    log_ratio_acc_t, is_ratio_sq_acc_t, loss_sum_t,
                ])
                dist.all_reduce(diag_t, op=dist.ReduceOp.SUM)
                is_ratio_acc_t, is_clip_acc_t = diag_t[0], diag_t[1]
                is_tok_acc_t = diag_t[2].to(torch.long)
                log_ratio_acc_t, is_ratio_sq_acc_t = diag_t[3], diag_t[4]
                loss_sum_t = diag_t[5]
            # ---- Phase F (ALL RANKS): clip + step on the global gradient; identical grads +
            # identical AdamW state keep every rank's params byte-identical afterward. ----
            t_optimizer_start = time.monotonic()
            grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            torch.cuda.synchronize()
            times["optimizer_clip"] = time.monotonic() - t_optimizer_start
            loss_sum = float(loss_sum_t.detach())
            is_ratio_acc = float(is_ratio_acc_t.detach())
            is_clip_acc = float(is_clip_acc_t.detach())
            is_tok_acc = int(is_tok_acc_t.detach())
            log_ratio_acc = float(log_ratio_acc_t.detach())
            is_ratio_sq_acc = float(is_ratio_sq_acc_t.detach())
            grad_norm = float(grad_norm_t.detach())
            times["train"] = time.monotonic() - t_train_start

            # ---- Phase F (debug guard, ALL RANKS): confirm the replicas stayed bit-identical.
            # Gated by `world_size > 1` (and the flag) so all ranks issue its 4 collectives or
            # none, in the all-ranks region; limited to step < 3 to bound overhead. See
            # assert_replica_sync for the placement constraint.
            if args.debug_assert_replica_sync and world_size > 1 and step < 3:
                assert_replica_sync(model, step)

            # ---- Phase G (ALL RANKS): single barrier strictly before rank 0's weight-sync
            # round-trip (no trainer-group collective may appear between MSG_SYNC_WEIGHTS and
            # MSG_WEIGHTS_READY below). ----
            if world_size > 1:
                dist.barrier()

            # ---- push fresh weights to the generators (RANK 0 ONLY) ----
            # The step's collectives are done, so workers are blocked at the NEXT step's Phase-C
            # broadcast (when one exists). A non-final weight-sync failure therefore broadcasts one
            # poison sentinel that workers consume at next-step Phase C; the final step has no
            # Phase-C receiver, so rank 0 just raises.
            if master_process:
                try:
                    _sync_weights_to_generators(times)
                    if (
                        args.inloop_eval_every > 0
                        and has_next_step
                        and step > 0
                        and step % args.inloop_eval_every == 0
                    ):
                        inloop_eval.launch_round(step, current_version)
                except BaseException:
                    if has_next_step:
                        _broadcast_poison(poison_sent)
                    raise
            times["step"] = time.monotonic() - t_step_start

            # ---- metrics (RANK 0 ONLY): the collect-side counts are rank-0-local; the
            # all-reduced global_token_count drives train throughput. Workers skip to the next
            # step's Phase-C scatter. ----
            if master_process:
                try:
                    _finish_step_rank0(
                        step, lr, cs, times, adv_all, loss_sum, grad_norm,
                        is_ratio_acc, is_clip_acc, is_tok_acc, log_ratio_acc, is_ratio_sq_acc,
                        global_token_count, num_microbatches,
                    )
                except BaseException:
                    if has_next_step:
                        _broadcast_poison(poison_sent)
                    raise
    finally:
        tensorize_pool.shutdown(wait=False, cancel_futures=True)
        if decode_pool is not None:
            decode_pool.shutdown(wait=False, cancel_futures=True)
        # RANK-0-ONLY teardown: only rank 0 owns the child/queues/wsync/wandb. Workers just
        # destroy the trainer process group. compute_cleanup() is a no-op when uninitialized.
        if master_process and control_q is not None:
            try:
                control_q.put({"type": MSG_SHUTDOWN})
            except Exception:
                pass
            if child is not None and child.pid is not None:
                child.join(timeout=30)
                if child.is_alive():
                    child.terminate()
        if master_process and wandb_rollouts_enabled and wandb_rollout_rows:
            import wandb
            wandb_run.log({"rollouts": wandb.Table(columns=wandb_rollout_columns, data=wandb_rollout_rows)})
        if master_process and final_rollout_buf:
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                final_path = run_dir / "final_rollouts.jsonl.gz"
                with gzip.open(final_path, "wt", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "type": "meta", "run": args.run, "train_data": str(args.train_data),
                        "steps": [rows[0]["step"] for rows in final_rollout_buf if rows],
                        "git_commit": _git_commit(),
                    }) + "\n")
                    for rows in final_rollout_buf:
                        for row in rows:
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                print0(f"Saved final {len(final_rollout_buf)}-step rollouts to {final_path}")
            except Exception as exc:
                print0(f"final-rollouts write failed (non-fatal): {exc!r}")
        if rollout_stream_q is not None:
            _stream_flush_pending()
            rollout_stream_q.put(None)            # writer drains FIFO, then exits
            rollout_stream_thread.join(timeout=30)  # bounded: don't hang teardown on a wedged write
        wandb_run.finish()
        run_logger.close()
        if world_size > 1:
            compute_cleanup()


# ============================ CLI ============================
def _vllm_flag(prefix: str, key: str) -> str:
    """CLI flag for one VLLM_TUNE_KEYS knob, e.g. ('eval_', 'max_num_seqs') -> '--eval-vllm-max-num-seqs'."""
    return f"--{prefix}vllm_{key}".replace("_", "-")


def _add_vllm_tuning_args(parser: argparse.ArgumentParser, *, prefix: str, ctx: str,
                          seqs_note: str, batched_note: str) -> None:
    """Register one --[eval-]vllm-* flag family (exactly the VLLM_TUNE_KEYS knobs, all
    defaulting to None = preserve vLLM's default; see vllm_tuning_from_args)."""
    flag = lambda key: _vllm_flag(prefix, key)  # noqa: E731
    parser.add_argument(flag("max_num_seqs"), type=int, default=None,
                        help=f"vLLM scheduler max_num_seqs for {ctx}. {seqs_note}")
    parser.add_argument(flag("max_num_batched_tokens"), type=int, default=None,
                        help=f"vLLM scheduler max_num_batched_tokens for {ctx}. {batched_note}")
    parser.add_argument(flag("max_num_scheduled_tokens"), type=int, default=None,
                        help=f"Optional vLLM scheduler max_num_scheduled_tokens for {ctx}. "
                             "Unset defaults to max_num_batched_tokens.")
    parser.add_argument(flag("max_num_partial_prefills"), type=int, default=None,
                        help=f"Optional vLLM chunked-prefill concurrency for {ctx}.")
    parser.add_argument(flag("max_long_partial_prefills"), type=int, default=None,
                        help=f"Optional vLLM max concurrent long partial prefills for {ctx}.")
    parser.add_argument(flag("long_prefill_token_threshold"), type=int, default=None,
                        help=f"Optional vLLM token threshold for treating a prefill as long ({ctx}).")
    parser.add_argument(flag("enable_chunked_prefill"), action=argparse.BooleanOptionalAction, default=None,
                        help=f"Optionally force vLLM chunked prefill on/off for {ctx}. "
                             "Unset preserves vLLM's default.")
    parser.add_argument(flag("scheduler_reserve_full_isl"), action=argparse.BooleanOptionalAction, default=None,
                        help=f"Optionally force vLLM scheduler_reserve_full_isl on/off for {ctx}. "
                             "Unset preserves vLLM's default.")
    parser.add_argument(flag("stream_interval"), type=int, default=None,
                        help=f"Optional vLLM streaming interval for {ctx}.")
    parser.add_argument(flag("performance_mode"), choices=["balanced", "interactivity", "throughput"],
                        default=None,
                        help=f"Optional vLLM performance_mode for {ctx}. Unset preserves vLLM's default.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full fine-tuning RL on fixed Reasoning Gym Sokoban JSONL data with a Hugging Face causal LM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="HF model id or local path")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders")
    parser.add_argument("--device", type=str, default="", help="cuda|cpu|mps, empty means autodetect")
    parser.add_argument("--train-data", type=Path, default=Path("datasets/sokoban_train.jsonl"), help="Train JSONL file with question, answer, and metadata.gamestr")
    parser.add_argument("--eval-data", type=Path, default=Path("datasets/sokoban_eval.jsonl"), help="Eval JSONL file with question, answer, and metadata.gamestr")
    parser.add_argument(
        "--verify-datasets-only",
        action="store_true",
        help="Validate train/eval JSONL files and gold-answer scores, then exit before loading a model or vLLM",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="Parameter dtype used when loading the model. LM head is kept in fp32 when non-fp32 is chosen.",
    )
    parser.add_argument("--run", type=str, default="dummy", help="W&B run name; 'dummy' disables W&B")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Base directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on optimization steps")
    parser.add_argument("--device-batch-size", type=int, default=8, help="Max forward/backward batch size during training")
    parser.add_argument("--examples-per-step", type=int, default=48, help="Sokoban puzzles per optimizer step")
    parser.add_argument("--num-samples", type=int, default=16, help="Samples per Sokoban puzzle")
    parser.add_argument("--max-new-tokens", type=int, default=6144,
                        help="Rollout generation budget per puzzle (README spec). Qwen3-4B runs with "
                             "thinking enabled, so it needs room to finish reasoning AND emit the answer; "
                             "too small => 100%% length-truncation => 0 reward => no training signal. With "
                             "--interruption this is the thinking budget before the forced-answer continuation.")
    parser.add_argument("--interruption", action=argparse.BooleanOptionalAction, default=False,
                        help="Interruption-based length control (ScaleRL A.10): when a rollout reaches "
                             "--max-new-tokens still thinking, append an end-of-thinking marker and force a "
                             "short final answer instead of hard-truncating to reward 0. The injected marker "
                             "tokens are masked out of the policy-gradient loss. Needs --max-model-len >= "
                             "prompt + max-new-tokens + marker + interrupt-answer-tokens.")
    parser.add_argument("--interrupt-answer-tokens", type=int, default=512,
                        help="Token budget for the forced final answer after an interruption.")
    parser.add_argument("--interrupt-min-tokens", type=int, default=None,
                        help="If set, randomly choose the phase-1 thinking cap per rollout group from "
                             "[--interrupt-min-tokens, --interrupt-max-tokens]. Default keeps a fixed "
                             "--max-new-tokens cap.")
    parser.add_argument("--interrupt-max-tokens", type=int, default=None,
                        help="Upper bound for randomized phase-1 interruption. Defaults to "
                             "--max-new-tokens when --interrupt-min-tokens is set.")
    parser.add_argument("--interrupt-temperature", type=float, default=None,
                        help="Sampling temperature for the forced final-answer continuation; "
                             "default inherits --temperature.")
    parser.add_argument("--interrupt-top-p", type=float, default=None,
                        help="Nucleus sampling threshold for the forced final-answer continuation; "
                             "default inherits --top-p.")
    parser.add_argument("--interrupt-top-k", type=int, default=None,
                        help="Top-k sampling for the forced final-answer continuation; "
                             "default inherits --top-k.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling; 0 disables")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling threshold; 0 disables")
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--optimizer", choices=["adamw"], default="adamw")
    parser.add_argument("--adam-eps", type=float, default=1e-15, help="AdamW epsilon.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--init-lr-frac", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=100, help="Linear LR warmup steps before the main schedule.")
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "linear"],
        default="constant",
        help="LR schedule after warmup.",
    )
    parser.add_argument(
        "--min-lr-frac",
        type=float,
        default=0.0,
        help="Floor for the linear schedule, as a fraction of peak LR. Decay stops here and HOLDS "
             "(0.0 = decay to zero, the legacy behavior).",
    )
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=None,
        help="Absolute step at which the linear decay reaches --min-lr-frac, after which LR holds at "
             "the floor. Defaults to num_steps (decay spans the whole run).",
    )
    parser.add_argument(
        "--lr-decay-start",
        type=int,
        default=None,
        help="Absolute step at which the linear decay BEGINS; LR holds at peak from the end of warmup "
             "until here (trapezoid). Defaults to the end of warmup (decay starts immediately).",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--loss-normalization",
        choices=["token", "sequence", "prompt"],
        default="sequence",
        help="Policy-gradient loss aggregation: 'sequence' (per-sequence mean then averaged over samples, "
             "GRPO sample-level — the default, stable in this regime), 'token' (per-token over the whole "
             "batch, DAPO), or 'prompt' (per-prompt-group token-average then averaged over prompts, ScaleRL §3.2 "
             "/ J_ScaleRL — only safe in a low-truncation regime).",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                        help="Trade compute for activation memory. On by default: the fp32 trainer GPU "
                             "(params+Adam+grad ~64GB) plus 6144-token rollouts otherwise OOMs.")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                        help="torch.compile the transformer body (dynamic shapes) to cut the fp32 train "
                             "step. Off by default; verify grad_norm/loss match eager and watch for "
                             "recompile churn (variable sequence lengths) before trusting it.")
    parser.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2", "eager"],
                        default="flash_attention_2",
                        help="HF attention implementation for the TRAINER body forward. Default "
                             "'flash_attention_2': measured ~15-30%% faster than sdpa for this workload "
                             "on H100 (~60 vs ~75-95 us/token). Served from the HF `kernels` hub without "
                             "a from-source flash-attn build (needs the `kernels` pkg + a one-time hub "
                             "download). If the kernel can't load it falls back to 'sdpa' with a warning. "
                             "fp32 head/tie is unaffected (FA2 runs only the bf16 body). Pass 'sdpa' to "
                             "force it instead.")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=60, help="Save every N steps; 0 disables periodic saves")
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--trim-after-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim generated sequences after the first reward-parseable Sokoban answer",
    )
    parser.add_argument("--vllm-gpu-mem-util", type=float, default=0.85,
                        help="vLLM gpu_memory_utilization per generator GPU")
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=False,
                        help="Pass enforce_eager=True to the rollout vLLM engine. This can reduce cold-start "
                             "CUDA graph capture time but may lower steady-state generation throughput; off by default.")
    _add_vllm_tuning_args(
        parser, prefix="", ctx="rollout generation",
        seqs_note="NOTE: when unset, the in-process AsyncLLM (ENGINE_CONTEXT) falls back to 128, "
                  "NOT the H100 1024 tier, so the RECIPE sets it explicitly. Re-clamped to "
                  "min(max_num_seqs, max_num_batched_tokens) — raise both together.",
        batched_note="NOTE: when unset, the in-process AsyncLLM (ENGINE_CONTEXT) falls back to "
                     "2048, NOT the H100 8192 tier, so the RECIPE sets it explicitly.",
    )
    parser.add_argument("--inflight-requests", type=int, default=16,
                        help="Outstanding puzzle-group budget (generating + awaiting file-order "
                             "consumption); tickets reissue as groups are consumed, so this also "
                             "bounds behavior staleness at inflight/groups-consumed-per-step")
    parser.add_argument("--max-staleness", type=int, default=4,
                        help="Drop rollouts older than this many weight versions (PipelineRL-k)")
    parser.add_argument("--logits-chunk-size", type=int, default=1024,
                        help="Sequence chunk size for the memory-efficient chunked LM-head policy loss. "
                             "Larger values reduce LM-head loop overhead but increase fp32 logits memory; "
                             "the Modal recipe uses 2048 with --device-batch-size 2.")
    parser.add_argument("--train-autocast-dtype", choices=["none", "bfloat16"], default="bfloat16",
                        help="Autocast dtype for the trainer's transformer-body forward; the LM head "
                             "stays fp32. 'none' runs the body in fp32 (more activation memory). bf16 "
                             "halves activation memory and matches the bf16 vLLM generator.")
    parser.add_argument("--debug-assert-replica-sync", action="store_true",
                        help="After each optimizer step (first few steps), assert all trainer-rank "
                             "model replicas are bit-identical via an all-reduce MIN==MAX checksum. "
                             "Debug guard; small overhead. No-op at world_size==1.")
    parser.add_argument("--loss-fn", choices=["cispo", "grpo", "gspo"], default="cispo",
                        help="Policy loss. 'cispo' (default, record 1): stop-grad clipped IS weight times "
                             "log-prob — clipped tokens keep a capped gradient. 'grpo': PPO-clip surrogate "
                             "(gradient through the ratio; clipped tokens get ZERO gradient), DAPO = grpo + "
                             "--clip-eps-high 0.28 + --loss-normalization token. 'gspo': length-normalized "
                             "SEQUENCE-level IS ratio, clipped (use with --loss-normalization sequence). "
                             "All ratios are vs the vLLM behavior logprobs (async off-policy, up to "
                             "--max-staleness versions stale; there is no second 'old policy' forward), so "
                             "the clip also absorbs the bf16-generator/fp32-trainer mismatch.")
    parser.add_argument("--cispo-eps", type=float, default=4.0,
                        help="CISPO upper clip epsilon_max for the importance weight")
    parser.add_argument("--clip-eps-low", type=float, default=None,
                        help="Lower clip epsilon for grpo/gspo (ratio floored at 1-eps_low). Default "
                             "resolves per loss: grpo 0.2 (PPO/GRPO), gspo 3e-4 (the GSPO paper scale — "
                             "the seq ratio is length-normalized, so eps is ~3 orders smaller).")
    parser.add_argument("--clip-eps-high", type=float, default=None,
                        help="Upper clip epsilon for grpo/gspo (ratio capped at 1+eps_high). Default "
                             "resolves per loss: grpo 0.2 (DAPO clip-higher: 0.28), gspo 4e-4.")
    parser.add_argument("--clip-ratio-c", type=float, default=3.0,
                        help="Dual-clip PPO constant c (grpo only): for negative advantages the objective "
                             "is floored at c*A so a large stale-token ratio can't dominate the gradient. "
                             "Must be > 1. <= 0 disables dual-clip.")
    parser.add_argument("--kl-tau", type=float, default=0.0,
                        help="Optional prime-rl-style mismatch-KL anchor: adds kl_tau * mean(log_ratio**2) "
                             "over valid tokens, aggregated with the SAME per-prompt/per-sample/per-token "
                             "denominator as the policy loss. Back-props through the policy logprobs only "
                             "(behavior detached). 0.0 (default) disables it (byte-identical). Active only "
                             "when behavior logprobs are present (the CISPO branch). prime-rl's default is 1e-3.")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="vLLM max_model_len (prompt + generation) for the generators; must cover "
                             "prompt (~700) + --max-new-tokens (6144).")
    parser.add_argument("--zero-variance-filter", action=argparse.BooleanOptionalAction, default=True,
                        help="Drop puzzle groups whose rewards are all equal (zero advantage). "
                             "Disable to keep them (e.g. plumbing smoke tests where the model gets no reward).")
    parser.add_argument("--save-rollouts", action=argparse.BooleanOptionalAction, default=True,
                        help="Stream every consumed rollout group (trained + zero-variance) to "
                             "<output-dir>/<run>/rollouts.jsonl.gz, appended by a background thread "
                             "every --rollout-flush-every steps. A prematurely stopped run keeps its "
                             "full history up to the last flush.")
    parser.add_argument("--wandb-rollout-samples", type=int, default=8,
                        help="Rollouts sampled per step (spread by reward) into a browsable W&B Table; 0 disables.")
    parser.add_argument("--final-rollout-steps", type=int, default=10,
                        help="Keep the last N steps' FULL rollouts in memory and write them to "
                             "<output-dir>/<run>/final_rollouts.jsonl.gz at run end (record artifact: "
                             "end-of-training health is auditable offline); 0 disables.")
    parser.add_argument("--rollout-flush-every", type=int, default=10,
                        help="Hand the pending rollout-stream rows to the background writer every N "
                             "steps (each flush = one gzip member appended to rollouts.jsonl.gz); "
                             "0 = flush only at teardown.")
    parser.add_argument("--eval-save-rollouts", action=argparse.BooleanOptionalAction, default=True,
                        help="With --eval-output/--eval-only: write every eval completion to "
                             "<eval-json-stem>.rollouts.jsonl.gz so the record can be re-scored "
                             "offline by verify_record.py.")

    parser.add_argument("--inloop-eval-every", type=int, default=0,
                        help="Run held-out eval through the shared vLLM engine every N training steps; "
                             "0 disables. Eval requests are excluded from training bookkeeping.")
    parser.add_argument("--inloop-eval-k", type=int, default=4,
                        help="Samples per held-out puzzle for in-loop eval.")
    parser.add_argument("--inloop-eval-max-tokens", type=int, default=6144,
                        help="Generation budget per puzzle for in-loop eval.")
    parser.add_argument("--inloop-eval-interruption", action=argparse.BooleanOptionalAction, default=False,
                        help="Use interruption-based answer forcing for in-loop held-out eval, matching "
                             "the training length-control protocol.")
    parser.add_argument("--inloop-eval-interrupt-answer-tokens", type=int, default=512,
                        help="Token budget for the forced final answer in interrupted in-loop eval.")
    parser.add_argument("--inloop-eval-seed", type=int, default=12345,
                        help="Fixed sampling seed for paired in-loop eval rounds.")
    parser.add_argument("--inloop-eval-deadline-steps", type=int, default=5,
                        help="Warn if an in-loop eval round is still incomplete after this many training steps; "
                             "partial eval rounds are never logged as eval/pass_at_1.")

    # Standalone held-out evaluation (the authoritative leaderboard metric). Runs in its own
    # process with a dedicated vLLM engine sized for the full rollout budget; no torchrun/DDP.
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate a checkpoint (or the base model) on the held-out set and exit.")
    parser.add_argument("--eval-checkpoint", type=Path, nargs="+", default=None,
                        help="Checkpoint dir(s) to evaluate; defaults to --model (evaluate the base model). "
                             "Pass one final checkpoint per training seed to run the full record eval: "
                             "each is evaluated under the same protocol, then the significance test "
                             "(mean pass@1 > --eval-target at --eval-alpha) prints a PASS/FAIL verdict.")
    parser.add_argument("--eval-target", type=float, default=0.50,
                        help="Leaderboard TARGET: the pass@1 a record must clear (see README rules).")
    parser.add_argument("--eval-alpha", type=float, default=0.01,
                        help="Significance level for the record t-test (multi-checkpoint eval).")
    parser.add_argument("--eval-k", type=int, default=16,
                        help="Samples per puzzle for --eval-only; k=16 enables pass@{1,4,8,16}.")
    parser.add_argument("--eval-max-tokens", type=int, default=6144,
                        help="Generation budget per puzzle for --eval-only (leaderboard protocol).")
    parser.add_argument("--eval-max-model-len", type=int, default=8192,
                        help="vLLM max_model_len for the eval engine; must be >= prompt + --eval-max-tokens.")
    parser.add_argument("--eval-interruption", action=argparse.BooleanOptionalAction, default=False,
                        help="Use interruption-based answer forcing for --eval-only.")
    parser.add_argument("--eval-interrupt-answer-tokens", type=int, default=512,
                        help="Token budget for the forced final answer in interrupted --eval-only.")
    parser.add_argument("--eval-interrupt-temperature", type=float, default=None,
                        help="Sampling temperature for forced final-answer continuations during eval; "
                             "default inherits --eval-temperature.")
    parser.add_argument("--eval-interrupt-top-p", type=float, default=None,
                        help="Nucleus sampling threshold for forced final-answer continuations during eval; "
                             "default inherits --eval-top-p.")
    parser.add_argument("--eval-interrupt-top-k", type=int, default=None,
                        help="Top-k sampling for forced final-answer continuations during eval; "
                             "default inherits --eval-top-k.")
    parser.add_argument("--eval-temperature", type=float, default=0.8, help="--eval-only sampling temperature.")
    parser.add_argument("--eval-top-p", type=float, default=0.95, help="--eval-only nucleus sampling threshold.")
    parser.add_argument("--eval-top-k", type=int, default=0, help="--eval-only top-k (0 disables).")
    parser.add_argument("--eval-seed", type=int, default=12345,
                        help="Sampling/bootstrap seed for --eval-only (vary per leaderboard seed; keep eval DATA fixed).")
    parser.add_argument("--eval-output", type=Path, default=None,
                        help="Per-run eval JSON path; default <output-dir>/<run>/eval_step<NNN>.json.")
    parser.add_argument("--eval-step", type=int, default=None,
                        help="Step number recorded in the eval JSON; parsed from the checkpoint dir name if omitted.")
    parser.add_argument("--eval-vllm-dp", type=int, default=1, help="Data-parallel GPUs for the eval engine.")
    parser.add_argument("--eval-gpu-mem-util", type=float, default=0.85,
                        help="gpu_memory_utilization for the eval engine.")
    parser.add_argument("--eval-vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=False,
                        help="Pass enforce_eager=True to the standalone eval vLLM engine.")
    _add_vllm_tuning_args(
        parser, prefix="eval_", ctx="the standalone eval engine",
        seqs_note="When unset, the AsyncLLM (ENGINE_CONTEXT) falls back to 128, NOT the H100 1024 "
                  "tier. To realize a higher value you must also raise --eval-concurrency in lockstep.",
        batched_note="When unset, the AsyncLLM (ENGINE_CONTEXT) falls back to 2048, NOT the H100 "
                     "8192 tier.",
    )
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Evaluate only the first N puzzles (smoke tests); default = full eval set.")
    parser.add_argument("--eval-concurrency", type=int, default=128,
                        help="Max concurrent in-flight eval requests (asyncio semaphore) for the "
                             "standalone --eval-only path. Raise in lockstep with --eval-vllm-max-num-seqs "
                             "so the engine is actually fed beyond 128 sequences.")
    return parser


def _validate_sampling_args(args: argparse.Namespace) -> None:
    for name in ("temperature", "interrupt_temperature", "eval_temperature", "eval_interrupt_temperature"):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("top_p", "interrupt_top_p", "eval_top_p", "eval_interrupt_top_p"):
        value = getattr(args, name)
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    for name in ("top_k", "interrupt_top_k", "eval_top_k", "eval_interrupt_top_k"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    _validate_vllm_scheduler_args(args)


_VLLM_POSITIVE_KEYS = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_num_scheduled_tokens",
    "max_num_partial_prefills",
    "max_long_partial_prefills",
    "stream_interval",
)


def _validate_vllm_scheduler_args(args: argparse.Namespace) -> None:
    for prefix in ("", "eval_"):
        for key in _VLLM_POSITIVE_KEYS:
            value = getattr(args, f"{prefix}vllm_{key}")
            if value is not None and value < 1:
                raise ValueError(f"{_vllm_flag(prefix, key)} must be positive")
        threshold = getattr(args, f"{prefix}vllm_long_prefill_token_threshold")
        if threshold is not None and threshold < 0:
            raise ValueError(f"{_vllm_flag(prefix, 'long_prefill_token_threshold')} must be non-negative")
        partial = getattr(args, f"{prefix}vllm_max_num_partial_prefills")
        long_partial = getattr(args, f"{prefix}vllm_max_long_partial_prefills")
        if (long_partial if long_partial is not None else 1) > (partial if partial is not None else 1):
            raise ValueError(f"{_vllm_flag(prefix, 'max_long_partial_prefills')} must be <= "
                             f"{_vllm_flag(prefix, 'max_num_partial_prefills')}")


# Training-pipeline integer-bound checks: (attr, minimum, requirement phrase). Bespoke checks
# (num_samples >= 2, init_lr_frac range, adam_eps > 0, the interrupt min/max relation) live in
# _validate_pipeline_args below.
_PIPELINE_MIN_ARGS = (
    ("device_batch_size", 1, "at least 1"),
    ("max_new_tokens", 1, "positive"),
    ("logits_chunk_size", 1, "positive"),
    ("interrupt_answer_tokens", 1, "positive"),
    ("warmup_steps", 0, "non-negative"),
    ("inloop_eval_every", 0, "non-negative"),
    ("inloop_eval_k", 1, "at least 1"),
    ("inloop_eval_max_tokens", 1, "positive"),
    ("inloop_eval_interrupt_answer_tokens", 1, "positive"),
    ("inloop_eval_deadline_steps", 1, "at least 1"),
    ("eval_max_tokens", 1, "positive"),
    ("eval_interrupt_answer_tokens", 1, "positive"),
)


def _resolve_loss_args(args: argparse.Namespace) -> None:
    """Resolve per-loss clip defaults (the PPO and GSPO epsilons differ by ~3 orders of
    magnitude, so a shared default would be a footgun) and validate the combination.
    Mutates args in place so the run-log header records the RESOLVED values."""
    if args.clip_ratio_c is not None and args.clip_ratio_c <= 0:
        args.clip_ratio_c = None  # documented disable switch
    if args.loss_fn == "cispo":
        if args.cispo_eps is None or args.cispo_eps <= 0:
            raise ValueError("--cispo-eps must be positive for --loss-fn cispo")
        return
    if args.loss_fn == "grpo":
        if args.clip_eps_low is None:
            args.clip_eps_low = 0.2
        if args.clip_eps_high is None:
            args.clip_eps_high = 0.2
        if args.clip_ratio_c is not None and args.clip_ratio_c <= 1.0:
            raise ValueError("--clip-ratio-c must be > 1 (or <= 0 to disable dual-clip)")
    elif args.loss_fn == "gspo":
        if args.clip_eps_low is None:
            args.clip_eps_low = 3e-4
        if args.clip_eps_high is None:
            args.clip_eps_high = 4e-4
        args.clip_ratio_c = None  # dual-clip is a grpo-only mechanism
    if not (0.0 < args.clip_eps_low < 1.0):
        raise ValueError("--clip-eps-low must be in (0, 1)")
    if args.clip_eps_high <= 0.0:
        raise ValueError("--clip-eps-high must be positive")


def _validate_pipeline_args(args: argparse.Namespace) -> None:
    """Bounds checks for a TRAINING launch (the eval-only/verify-only paths skip these)."""
    for attr, minimum, requirement in _PIPELINE_MIN_ARGS:
        if getattr(args, attr) < minimum:
            raise ValueError(f"--{attr.replace('_', '-')} must be {requirement}")
    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2 so per-puzzle advantages are non-zero")
    if not (0.0 < args.init_lr_frac <= 1.0):
        raise ValueError("--init-lr-frac must be in (0, 1]")
    if args.adam_eps <= 0:
        raise ValueError("--adam-eps must be positive")
    if args.interrupt_min_tokens is not None or args.interrupt_max_tokens is not None:
        if not args.interruption:
            raise ValueError("--interrupt-min-tokens/--interrupt-max-tokens require --interruption")
        interrupt_min = (
            args.interrupt_min_tokens
            if args.interrupt_min_tokens is not None
            else args.interrupt_max_tokens
        )
        interrupt_max = (
            args.interrupt_max_tokens
            if args.interrupt_max_tokens is not None
            else args.max_new_tokens
        )
        if interrupt_min < 1:
            raise ValueError("--interrupt-min-tokens must be positive")
        if interrupt_max < 1:
            raise ValueError("--interrupt-max-tokens must be positive")
        if interrupt_min > interrupt_max:
            raise ValueError("--interrupt-min-tokens must be <= --interrupt-max-tokens")
        if interrupt_max > args.max_new_tokens:
            raise ValueError("--interrupt-max-tokens must be <= --max-new-tokens")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    cli = sys.argv[1:] if argv is None else argv
    # Prepend the hardcoded RECIPE (top of file) so the benchmark recipe is the baseline; any flag
    # passed on the CLI comes AFTER and overrides its recipe value (argparse last-wins).
    args = parser.parse_args([*RECIPE, *cli])
    _validate_sampling_args(args)
    _resolve_loss_args(args)

    if args.eval_only:
        run_standalone_eval(args)
        return

    if args.verify_datasets_only:
        train_task = load_sokoban_jsonl_dataset(
            args.train_data,
            split_name="train",
            verify_gold=True,
        )
        eval_task = load_sokoban_jsonl_dataset(
            args.eval_data,
            split_name="eval",
            verify_gold=True,
        )
        print(
            f"Dataset verification passed: train={len(train_task)} ({args.train_data}), "
            f"eval={len(eval_task)} ({args.eval_data})",
            flush=True,
        )
        return

    _validate_pipeline_args(args)

    train_task = load_sokoban_jsonl_dataset(
        args.train_data,
        split_name="train",
        verify_gold=False,
    )
    eval_task = load_sokoban_jsonl_dataset(
        args.eval_data,
        split_name="eval",
        verify_gold=False,
    )

    run_pipeline(args, train_task=train_task, eval_task=eval_task)


if __name__ == "__main__":
    main()
