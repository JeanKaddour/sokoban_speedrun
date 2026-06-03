"""
Sokoban Speedrun
Run it directly (`python speedrun.py ...`), as a module (`python -m speedrun ...`), or under torchrun
for the multi-trainer data-parallel pipeline. The vLLM child process produces
rollouts on-policy while a CISPO trainer learns from them:
- sample several completions per puzzle, reward Sokoban solutions verified by Reasoning Gym
- advantage = reward minus per-puzzle mean, trained only on generated tokens
- CISPO importance weighting corrects for off-policy / stale rollouts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import reasoning_gym
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


IGNORE_INDEX = -100
NODE_GPUS = 8  # the speedrun targets one 8xH100 node; trainers + vLLM generators must fit within it
WEIGHT_SYNC_TIMEOUT_S = 300.0  # bound the rank-0 weight broadcast so a dead vLLM child can't hang the trainer
ROLLOUT_GET_TIMEOUT_S = 300.0  # bound result-queue waits so a silent vLLM child crash can't stall the trainer
MSG_GENERATE = "generate"
MSG_ROLLOUT = "rollout"
MSG_INIT_WEIGHTS = "init_weights"
MSG_SYNC_WEIGHTS = "sync_weights"
MSG_SHUTDOWN = "shutdown"
MSG_ENGINE_READY = "engine_ready"
MSG_WEIGHTS_READY = "weights_ready"
MSG_ERROR = "error"
ANSWER_COMPLETE_RE = re.compile(r"####\s*[UDLRudlr]+(?=$|[^UDLRudlr])")
ANSWER_TAG_RE = re.compile(r"<answer>.*?</answer>", re.IGNORECASE | re.DOTALL)
_BOARD_PLACEHOLDER = "{board}"
SOKOBAN_RG_PROMPT_TEMPLATE = """You are solving a Sokoban puzzle. You are the player (*); push every box (@) onto a goal (X).

The board is shown with 0-indexed row numbers down the left side and column numbers across the top, so every cell has an address (row, col). Rows increase downward and columns increase rightward.

Legend:
* - The player
% - The player on a goal
@ - A box
X - A goal
$ - A box already on a goal (leave it there)
+ - A wall
- - An empty position

Moves (each steps the player one cell):
- U = up (row - 1), D = down (row + 1), L = left (col - 1), R = right (col + 1).

Rules:
- You may step onto empty cells (-) and goals (X), but never onto a wall (+).
- To push a box, step toward it and the box slides one cell in the same direction. The cell beyond the box must be empty or a goal.
- You cannot push a box into a wall or into another box, and you can never pull a box.
- Some boxes ($) already sit on goals; the board may start partly solved. Keep those boxes on their goals.
- The puzzle is solved when every box sits on a goal (no @ remain).

A blocked or off-board move is simply ignored, and extra moves do no harm, so prefer a sequence you are confident in and give your answer as soon as you have one.

Solve it step by step:
1. List the player position, every box (@) position, and every goal (X) position as (row, col).
2. For each box not already on a goal, plan how to reach the cell that lets you push it onto a goal, choosing an order that does not trap another box in a corner or against a wall.
3. Write out the full move string and simulate it in your head to confirm every box ends on a goal.

End with exactly one final line that wraps your move string in answer tags, like `<answer>...</answer>` — put only your U/D/L/R moves between the tags.

Here is your puzzle:
{board}
"""


def build_async_engine(
    model: str,
    num_dp: int,
    *,
    dtype: str = "bfloat16",
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    seed: int = 0,
    enforce_eager: bool = False,
    max_num_seqs: int = 512,
    max_num_batched_tokens: int = 8192,
):
    """Construct the in-process vLLM AsyncLLM with native NCCL weight transfer enabled."""
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    from vllm.config import WeightTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=model,
        tensor_parallel_size=1,
        data_parallel_size=num_dp,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        enforce_eager=enforce_eager,
        distributed_executor_backend=None,
        max_model_len=max_model_len,
        enable_log_requests=False,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        seed=seed,
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
    )
    return AsyncLLM.from_engine_args(engine_args)


def extract_sampled_logprobs(completion_output) -> list[float]:
    """Per-token logprob of the sampled token for vLLM completion outputs."""
    co = completion_output
    if co.logprobs is None:
        return []
    return [co.logprobs[i][tok_id].logprob for i, tok_id in enumerate(co.token_ids)]


async def generate_group(engine, prompt_token_ids: list[int], num_samples: int, sampling: dict) -> list[dict]:
    """Generate num_samples completions for one prompt."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    sp = SamplingParams(
        n=num_samples,
        temperature=sampling.get("temperature", 0.8),
        top_p=sampling.get("top_p", 1.0),
        top_k=sampling.get("top_k", 0),
        max_tokens=sampling.get("max_tokens", 1024),
        logprobs=sampling.get("logprobs", 0),
        seed=sampling.get("seed"),
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    prompt = TokensPrompt(prompt_token_ids=list(prompt_token_ids))
    final = None
    async for out in engine.generate(prompt, sp, str(uuid.uuid4())):
        final = out
    assert final is not None and final.finished
    # vLLM 0.22 can occasionally surface a None/empty completion in final.outputs (e.g. an
    # aborted sample); drop them so one bad sample can't crash the whole run. A fully-empty
    # group returns [] and is dropped downstream as zero-variance.
    outputs = [co for co in final.outputs if co is not None and co.token_ids is not None]
    dropped = len(final.outputs) - len(outputs)
    if dropped:
        print(f"[generate_group] dropped {dropped}/{len(final.outputs)} None/empty vLLM completion(s)", flush=True)
    return [
        {
            "token_ids": list(co.token_ids),
            "logprobs": extract_sampled_logprobs(co),
            "finish_reason": co.finish_reason,
        }
        for co in outputs
    ]


def _safe_get(q):
    """Non-throwing queue get with a short timeout; returns None on empty."""
    import queue

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
    from dataclasses import asdict

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


class DummyWandb:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


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


def should_save_checkpoint_for_step(step: int, num_steps: int, save_every: int, save_final: bool) -> bool:
    final_step = step == num_steps - 1
    periodic_save = save_every > 0 and step > 0 and step % save_every == 0
    return periodic_save or (final_step and save_final)


def _pipeline_has_next_step(step: int, num_steps: int) -> bool:
    return step < num_steps - 1


def build_optimizer(parameters: Iterator[torch.nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer != "adamw":
        raise ValueError(f"Unsupported optimizer {args.optimizer!r}; only 'adamw' is implemented")
    return torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )


def make_pad_example(pad_token_id: int) -> "RolloutExample":
    """A RolloutExample that pads a step batch to a multiple of the trainer world size, so
    every rank gets equal microbatches and none stalls in the all-reduce. It contributes zero
    to both loss numerator and token-count denominator: advantage 0.0 zeroes the gradient,
    prefix_length == length means no generated tokens, and empty behavior_logprobs keeps it out
    of both counts (see local_and_global_counts). The 2-token sequence satisfies
    make_rl_batch_varprefix's max_length >= 2 guard.
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


def pad_shards_to_equal(
    step_examples: list["RolloutExample"],
    world_size: int,
    pad_example: "RolloutExample",
) -> list[list["RolloutExample"]]:
    """Split the full step batch into `world_size` EQUAL-length shards, appending
    copies of `pad_example` to reach the next multiple of world_size. Equal microbatch
    counts per rank avoid stalls in the cross-rank grad all-reduce; pads contribute
    zero gradient (see make_pad_example). world_size == 1 returns [step_examples] as-is.
    """
    if world_size < 1:
        raise ValueError("world_size must be at least 1")
    if pad_example.advantage != 0.0:
        raise ValueError("pad_example must have advantage == 0.0 (zero gradient)")
    if world_size == 1:
        return [step_examples]

    total = len(step_examples)
    remainder = total % world_size
    pad_count = 0 if remainder == 0 else world_size - remainder
    # Padding is at most world_size - 1; it never approaches the per-rank shard size.
    assert pad_count < world_size, f"pad_count {pad_count} must be < world_size {world_size}"

    padded = list(step_examples) + [pad_example] * pad_count
    per_rank = len(padded) // world_size
    return [padded[r * per_rank:(r + 1) * per_rank] for r in range(world_size)]


def local_and_global_counts(local_examples: list["RolloutExample"]) -> tuple[int, int]:
    """Per-shard (sample_count, token_count) for one trainer rank, counting only real
    examples. Pad examples (empty behavior_logprobs) add 0 to BOTH counts, so summing
    across ranks recovers the unpadded global counts and the loss stays byte-equivalent
    to the unpadded full-batch gradient in both normalization branches.
    """
    local_sample_count = sum(1 for e in local_examples if e.behavior_logprobs)
    local_token_count = sum(len(e.behavior_logprobs) for e in local_examples)
    return local_sample_count, local_token_count


def _scatter_payload(
    padded_batch: list["RolloutExample"],
    adv_padded: list[float] | None,
    abort: bool = False,
) -> list:
    """The object rank 0 broadcasts to every trainer rank each step:
    [control dict with abort flag, full padded step batch, full-batch advantages].

    Each rank reads element 0 first, then either raises (abort) or slices its shard.
    Putting the abort flag first gives a poison sentinel and a real scatter the same
    wire shape, so a rank-0 Phase-A failure unblocks workers in lockstep.
    """
    return [{"abort": bool(abort)}, padded_batch, adv_padded]


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
    chunked_cispo_loss reaches into model.model / model.lm_head directly). Each rank runs its
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


def sanitize_run_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe or "run"


def set_seed(seed: int, device: torch.device | str | None = None) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


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


def build_sokoban_prompt(question: str, prompt_style: str = "rg") -> str:
    """Render a Sokoban prompt for the selected style."""
    board = extract_sokoban_board(question)
    if prompt_style == "nanochat":
        return board
    if prompt_style == "rg":
        return render_sokoban_rg_prompt(board)
    final_instruction = (
        "After your reasoning, end with exactly one final line: #### <moves>. "
        "The moves must use only U, D, L, and R."
    )
    if prompt_style == "sokoban":
        return (
            f"Here is your puzzle:\n{board}\n\n"
            "Reason about the board before answering. Do not restate these instructions. "
            f"{final_instruction}"
        )
    if prompt_style == "brief":
        return f"Here is your puzzle:\n{board}\n\nReason briefly about the puzzle, then answer. {final_instruction}"
    if prompt_style == "reason":
        return (
            f"Here is your puzzle:\n{board}\n\n"
            "First reason step by step about the player position, boxes, goals, and "
            "legal pushes. Then provide the final move string. "
            f"{final_instruction}"
        )
    if prompt_style == "instruct":
        return (
            "Solve the Sokoban puzzle. Reason about the board first, then put the "
            "final move string after '####'. Use only U, D, L, and R in the final "
            "answer.\n\n"
            f"Puzzle:\n{board}"
        )
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


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


def extract_sokoban_answer(completion: str) -> str | None:
    """Extract normalized Sokoban moves from answer tags or after the first #### marker."""
    tag = SOKOBAN_ANSWER_TAG_RE.search(completion)
    if tag is not None:
        return normalize_sokoban_moves(tag.group(1))

    marker = SOKOBAN_MARKER_RE.search(completion)
    if marker is None:
        return None
    for line in completion[marker.end() :].splitlines():
        if not line.strip():
            continue
        return normalize_sokoban_moves(line)
    return None


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


def encode_prompt(
    tokenizer,
    question: str,
    enable_thinking: bool = True,
    prompt_style: str = "rg",
) -> torch.Tensor:
    prompt = build_sokoban_prompt(question, prompt_style)
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


def decode_completion(tokenizer, sequence: torch.Tensor, prefix_length: int) -> str:
    generated = sequence[prefix_length:]
    return tokenizer.decode(generated.tolist(), skip_special_tokens=True)


def find_sokoban_answer_end(text: str) -> int | None:
    marker_match = ANSWER_COMPLETE_RE.search(text)
    tag_match = ANSWER_TAG_RE.search(text)
    matches = [match for match in (marker_match, tag_match) if match is not None]
    for match in sorted(matches, key=lambda candidate: candidate.start()):
        if extract_sokoban_answer(text[: match.end()]) is not None:
            return match.end()
    return None


def trim_sequence_after_answer(tokenizer, sequence: torch.Tensor, prefix_length: int) -> torch.Tensor:
    generated_ids = sequence[prefix_length:].tolist()
    if not generated_ids:
        return sequence
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    answer_end = find_sokoban_answer_end(full_text)
    if answer_end is None:
        return sequence
    target_text = full_text[:answer_end]
    target_len = len(target_text)

    # Decode is prefix-preserving on standard tokenizers — len(decode([:end])) is
    # non-decreasing in `end` — so we binary-search the smallest end that covers
    # the answer marker. O(log N) decodes instead of O(N).
    lo, hi = 1, len(generated_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        text = tokenizer.decode(generated_ids[:mid], skip_special_tokens=True)
        if len(text) >= target_len:
            hi = mid
        else:
            lo = mid + 1
    final_text = tokenizer.decode(generated_ids[:lo], skip_special_tokens=True)
    if final_text[:target_len] == target_text:
        return sequence[: prefix_length + lo]
    return sequence


def trim_generated_with_logprobs(tokenizer, sequence: torch.Tensor, prefix_length: int,
                                 behavior_logprobs: list[float]) -> tuple[torch.Tensor, list[float]]:
    """Trim a full sequence after its first parseable answer and truncate the behavior
    logprobs to match the kept generated tokens."""
    trimmed = trim_sequence_after_answer(tokenizer, sequence, prefix_length)
    kept_gen = int(trimmed.numel()) - prefix_length
    return trimmed, behavior_logprobs[:kept_gen]


def make_rl_batch_varprefix(sequences, prefixes, pad_token_id, device):
    if not sequences:
        raise ValueError("Cannot create an RL batch from zero sequences")
    if len(sequences) != len(prefixes):
        raise ValueError("sequences and prefixes must align")
    max_length = max(int(seq.numel()) for seq in sequences)
    if max_length < 2:
        raise ValueError("Sequences must contain at least two tokens")
    ids = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long, device=device)
    real = torch.zeros_like(ids, dtype=torch.bool)
    gen = torch.zeros_like(ids, dtype=torch.bool)
    for row, (seq, pfx) in enumerate(zip(sequences, prefixes)):
        seq = seq.to(device=device, dtype=torch.long)
        n = int(seq.numel())
        ids[row, :n] = seq
        real[row, :n] = True
        if n > pfx:
            gen[row, pfx:n] = True
    input_ids = ids[:, :-1]
    labels = ids[:, 1:].clone().masked_fill(~gen[:, 1:], IGNORE_INDEX)
    return input_ids, real[:, :-1], labels


def build_behavior_logprob_tensor_varprefix(sequences, behavior_logprobs, prefixes, device):
    if not (len(sequences) == len(behavior_logprobs) == len(prefixes)):
        raise ValueError("sequences, behavior_logprobs, and prefixes must align")
    max_length = max(int(seq.numel()) for seq in sequences)
    out = torch.zeros((len(sequences), max_length - 1), dtype=torch.float32, device=device)
    for row, (seq, blp, pfx) in enumerate(zip(sequences, behavior_logprobs, prefixes)):
        gen_len = int(seq.numel()) - pfx
        if gen_len <= 0:
            continue
        if len(blp) != gen_len:
            raise ValueError(f"row {row}: {len(blp)} logprobs for {gen_len} generated tokens")
        start = pfx - 1
        out[row, start:start + gen_len] = torch.tensor(blp, dtype=torch.float32, device=device)
    return out


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


def policy_gradient_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    valid_token_normalizer: torch.Tensor | float | None = None,
    valid_sample_normalizer: torch.Tensor | float | None = None,
    normalizer: float = 1.0,
    sequence_normalize: bool = False,
    behavior_logprobs: torch.Tensor | None = None,
    cispo_eps: float | None = None,
    stats: dict | None = None,
) -> torch.Tensor:
    token_logprobs = token_logprobs_from_logits(logits, labels)
    return policy_gradient_loss_from_token_logprobs(
        token_logprobs, labels, advantages, valid_token_normalizer, valid_sample_normalizer,
        normalizer, sequence_normalize, behavior_logprobs, cispo_eps, stats)


def policy_gradient_loss_from_token_logprobs(
    token_logprobs: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    valid_token_normalizer: torch.Tensor | float | None = None,
    valid_sample_normalizer: torch.Tensor | float | None = None,
    normalizer: float = 1.0,
    sequence_normalize: bool = False,
    behavior_logprobs: torch.Tensor | None = None,
    cispo_eps: float | None = None,
    stats: dict | None = None,
) -> torch.Tensor:
    advantage_weight = advantages.to(token_logprobs.dtype).unsqueeze(-1)  # (B, 1)
    if behavior_logprobs is not None:
        if cispo_eps is None or cispo_eps <= 0:
            raise ValueError("cispo_eps must be a positive float when behavior_logprobs is provided")
        # CISPO: stop-grad clipped importance weight min(rho, eps_max) multiplies the
        # advantage; only log pi_theta carries the gradient. Clamp in log-space before
        # exp to avoid overflow: exp(min(log_ratio, log eps)) == min(ratio, eps).
        log_ratio = token_logprobs - behavior_logprobs.to(token_logprobs.dtype)
        is_weight = torch.exp(torch.clamp(log_ratio, max=math.log(cispo_eps))).detach()
        token_weight = is_weight * advantage_weight  # (B, T)
        if stats is not None:
            # Off-policy drift diagnostics over valid (generated) tokens.
            with torch.no_grad():
                valid = labels != IGNORE_INDEX
                n_valid = valid.sum().clamp(min=1)
                raw_ratio = torch.exp(torch.clamp(log_ratio, max=20.0))
                stats["is_ratio_mean"] = float((raw_ratio * valid).sum() / n_valid)
                stats["is_clipped_frac"] = float(
                    ((log_ratio > math.log(cispo_eps)) & valid).sum() / n_valid
                )
                stats["valid_tokens"] = int(valid.sum())
    else:
        token_weight = advantage_weight              # (B, 1) broadcasts
    weighted_logprobs = token_logprobs * token_weight
    if sequence_normalize:
        valid_mask = labels != IGNORE_INDEX
        sample_lengths = valid_mask.sum(dim=-1).clamp(min=1).to(token_logprobs.dtype)
        sample_normalizer = labels.size(0) if valid_sample_normalizer is None else valid_sample_normalizer
        pg_objective = (weighted_logprobs.sum(dim=-1) / sample_lengths).sum() / (sample_normalizer * normalizer)
        return -pg_objective
    valid_count = (labels != IGNORE_INDEX).sum().clamp(min=1)
    token_normalizer = valid_count if valid_token_normalizer is None else valid_token_normalizer
    pg_objective = weighted_logprobs.sum() / (token_normalizer * normalizer)
    return -pg_objective


def chunked_cispo_loss(model, input_ids, attention_mask, labels, advantages, *,
                       behavior_logprobs=None, cispo_eps=None, chunk_size=1024,
                       valid_token_normalizer=None, valid_sample_normalizer=None,
                       normalizer=1.0, sequence_normalize=False, stats=None,
                       autocast_dtype=None):
    """Memory-efficient CISPO loss: run the base model to hidden states (small: B,T,H),
    then apply the LM head + cross-entropy in checkpointed sequence chunks to avoid
    materializing the full (T, vocab) fp32 logits. The head always runs in fp32; with
    autocast_dtype set, only the body forward is bf16 (halving activation memory). In the
    fp32 path (autocast_dtype=None) this equals
    policy_gradient_loss_from_logits(model(input_ids,...).logits, ...)."""
    base = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if base is None or lm_head is None:
        raise AttributeError("chunked_cispo_loss requires model.model + model.lm_head (HF CausalLM)")
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
        behavior_logprobs=behavior_logprobs, cispo_eps=cispo_eps, stats=stats)


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

    def metadata(self):
        # Param names/dtypes/shapes are invariant across training steps, so build the
        # lists once and reuse — this runs on the per-step rank-0 weight-sync path.
        if self._metadata is None:
            names, dtype_names, shapes = [], [], []
            for name, p in self.model.named_parameters():
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
            iterator=self.model.named_parameters(),
            trainer_args=NCCLTrainerSendWeightsArgs(group=self._group, packed=True),
        )


def task_reward(task, conversation, completion) -> float:
    return float(task.reward(conversation, completion))


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
    prompt_style: str = "rg",
    enable_thinking: bool = True,
    indices: list[int] | None = None,
    pass_at_ks: tuple[int, ...] = (1, 4, 8, 16),
    concurrency: int = 128,
    progress_every: int = 200,
) -> dict:
    """Evaluate the policy served by `engine` on `eval_task`: k samples per puzzle, each scored
    with reasoning_gym's binary solve check via `eval_task.evaluate`. Returns pass@1, unbiased
    pass@k, per-puzzle solve fractions and a confidence interval. Pure measurement (no logprobs,
    no training); reuses the same prompt/scoring path as the trainer for an apples-to-apples set."""
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
        prompt_ids = encode_prompt(tokenizer, conv["question"], enable_thinking, prompt_style)
        async with sem:
            samples = await generate_group(engine, prompt_ids.tolist(), k, eval_sampling)
        c = extract_fail = length_trunc = 0
        for s in samples:
            completion = tokenizer.decode(s["token_ids"], skip_special_tokens=True)
            if extract_sokoban_answer(completion) is None:
                extract_fail += 1
            if s.get("finish_reason") == "length":
                length_trunc += 1
            c += eval_task.evaluate(conv, completion)
        done += 1
        if progress_every and done % progress_every == 0:
            print(f"  eval progress: {done}/{n_total} puzzles", flush=True)
        return {"n": len(samples), "c": c, "extract_fail": extract_fail, "length_trunc": length_trunc}

    results = await asyncio.gather(*[_one(i) for i in indices])

    per_puzzle = [r["c"] / r["n"] if r["n"] else 0.0 for r in results]
    n_puzzles = len(per_puzzle)
    total_samples = sum(r["n"] for r in results)
    total_solved = sum(r["c"] for r in results)
    pass_at_1 = sum(per_puzzle) / max(1, n_puzzles)  # puzzle-weighted (== solved/total when n_i==k)

    pass_at_k = {
        j: sum(_pass_at_k_unbiased(r["n"], r["c"], j) for r in results) / max(1, n_puzzles)
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
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "n_extract_fail": sum(r["extract_fail"] for r in results),
        "n_length_trunc": sum(r["length_trunc"] for r in results),
        "sampling": eval_sampling,
    }


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


def run_standalone_eval(args: argparse.Namespace) -> None:
    """Authoritative held-out evaluation of a checkpoint (or the base model), decoupled from
    training: builds its own vLLM engine sized for the full rollout budget, runs the leaderboard
    pass@1/pass@k protocol on the fixed eval set, and writes a per-run JSON. No torchrun/DDP."""
    eval_task = load_sokoban_jsonl_dataset(args.eval_data, split_name="eval")
    model_path = str(args.eval_checkpoint) if args.eval_checkpoint else args.model

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

    print(
        f"[eval] model={model_path} eval_data={args.eval_data} "
        f"n={len(indices) if indices is not None else len(eval_task)} k={args.eval_k} "
        f"max_tokens={args.eval_max_tokens} sampling=temp{args.eval_temperature}/top_p{args.eval_top_p}/"
        f"top_k{args.eval_top_k}/seed{args.eval_seed}",
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
        )
        try:
            return await run_held_out_eval(
                engine, tokenizer, eval_task,
                k=args.eval_k, sampling=sampling,
                prompt_style=args.prompt_style, enable_thinking=args.enable_thinking,
                indices=indices,
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
            "n_extract_fail", "n_length_trunc", "sampling", "per_puzzle_solve_frac",
        )},
    }

    if args.eval_output is not None:
        out_path = Path(args.eval_output)
    else:
        run_name = args.run if (args.run and args.run != "dummy") else "eval"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name)
        suffix = f"step{step:06d}" if step is not None else "latest"
        out_path = args.output_dir / safe / f"eval_{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")

    pk = " ".join(f"pass@{j}={result['pass_at_k'][j]:.4f}" for j in sorted(result["pass_at_k"]))
    print(
        f"[eval] {model_path} | n={result['n_puzzles']} k={result['k']} | "
        f"pass@1={result['pass_at_1']:.4f} (95% CI [{result['ci_low']:.4f}, {result['ci_high']:.4f}], "
        f"se={result['se']:.4f}) | {pk} | "
        f"extract_fail={result['n_extract_fail']} length_trunc={result['n_length_trunc']} | -> {out_path}",
        flush=True,
    )


def _await_status(status_q, expected_type, timeout: float = 600.0):
    import time as _time
    deadline = _time.time() + timeout
    while _time.time() < deadline:
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


def run_pipeline(
    args: argparse.Namespace,
    train_task: FixedSokobanDataset | None = None,
    eval_task: FixedSokobanDataset | None = None,
) -> None:
    import multiprocessing as mp
    import queue

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
        try:
            dist.init_process_group(backend="nccl", device_id=device)
        except TypeError:
            dist.init_process_group(backend="nccl")
        dist.barrier()  # (1) the ONLY init barrier; symmetric all-rank; see invariant above.
    print0(f"Pipeline trainer world size: {world_size}")

    set_seed(args.seed, device)
    torch.set_float32_matmul_precision("high")

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
    sampling = dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    max_tokens=args.max_new_tokens)
    wandb_run = DummyWandb()
    model_perf_info = None
    rollout_fh = None

    def enqueue_puzzle(pid):
        conv = train_task[pid % len(train_task)]
        prompt_ids = encode_prompt(tokenizer, conv["question"], args.enable_thinking, args.prompt_style)
        puzzles[pid] = {"conversation": conv, "prompt_ids": prompt_ids, "prefix_length": int(prompt_ids.numel())}
        prompt_q.put({"type": MSG_GENERATE, "puzzle_id": pid,
                      "prompt_token_ids": prompt_ids.tolist(), "num_samples": args.num_samples,
                      "sampling": sampling})

    def _rank0_setup() -> None:
        nonlocal child, wsync, prompt_q, result_q, control_q, status_q
        nonlocal next_puzzle_id, wandb_run, model_perf_info, rollout_fh
        # Rendezvous steps (2)-(4); see the two-NCCL-group invariant above. All of this runs on
        # rank 0 only, with NO dist.* collective between (2) and (4): the workers are parked at the
        # (5) init-status broadcast, so a stray trainer-group collective here would have no peer
        # and hang the run.
        #
        # --- (2) spawn the vLLM child on GPUs T..T+M-1 ---
        layout = pipeline_trainer_layout(ddp_world_size, vllm_dp)
        ctx = mp.get_context("spawn")
        prompt_q, result_q, control_q, status_q = ctx.Queue(), ctx.Queue(), ctx.Queue(maxsize=0), ctx.Queue()
        visible = layout.visible_gpus
        cfg = dict(
            model=args.model, num_dp=vllm_dp,
            dtype="bfloat16",  # vLLM generates in bf16; the trainer may be fp32 (CISPO + FP32 logits head absorb the mismatch)
            max_model_len=args.max_model_len, gpu_memory_utilization=args.vllm_gpu_mem_util,
            seed=args.seed, inflight=args.inflight_requests, visible_gpus=visible, enforce_eager=False,
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
            enqueue_puzzle(next_puzzle_id); next_puzzle_id += 1

        # wandb / rollout files / model-perf are RANK-0-ONLY (only rank 0 logs metrics and
        # owns the collect loop). This is part of the guarded rank-0 init path so workers learn
        # about W&B or filesystem setup failures before waiting for the first step scatter.
        if args.run != "dummy":
            import wandb
            wandb_run = wandb.init(project="nanochat-rl-hf", name=args.run, config=vars(args))
        model_perf_info = collect_model_perf_info(model)
        if args.save_rollouts:
            run_dir.mkdir(parents=True, exist_ok=True)
            rollout_fh = open(run_dir / "rollouts.jsonl", "a", encoding="utf-8")
            print0(f"Saving rollouts to {run_dir / 'rollouts.jsonl'}")
    wandb_rollouts_enabled = False
    wandb_rollout_rows: list[list] = []
    wandb_rollout_columns = ["step", "weight_version", "staleness", "puzzle_id", "status",
                             "reward", "advantage", "gen_tokens", "finish_reason", "completion"]

    def _safe_qsize(q) -> float:
        try:
            return float(q.qsize())
        except (NotImplementedError, OSError):
            return -1.0

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

        for step in range(num_steps):
            has_next_step = _pipeline_has_next_step(step, num_steps)
            t_step_start = time.monotonic()
            lr = args.learning_rate * get_lr_multiplier(step, num_steps, args.init_lr_frac,
                                                        args.warmup_steps, args.lr_schedule)
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

            # ---- Phase A (RANK 0 ONLY): collect one step's worth of fresh, signal-bearing rollouts ----
            step_examples: list[RolloutExample] = []
            finish_reasons: list[str | None] = []
            staleness: list[int] = []
            group_solved: list[float] = []      # 1.0 if any sample in the group solved
            # Unbiased online proxy: counted over ALL fresh (non-stale) generated groups,
            # BEFORE the zero-variance filter that biases reward/solved_frac & group_pass_at_k.
            groups_seen_total = 0
            samples_seen_total = 0
            samples_solved_total = 0
            groups_any_solved_total = 0
            gen_tokens_total = 0
            prefill_tokens_total = 0
            puzzles_used = 0
            groups_zero_variance = 0
            groups_stale = 0
            consecutive_rejected = 0
            reject_limit = max(256, args.examples_per_step * 64)
            t_collect = 0.0
            adv_all = None
            shards: list[list[RolloutExample]] | None = None
            adv_shards: list[list[float]] | None = None
            if master_process:
                # Blanket handler: any exception in rank 0's Phase A/B — explicit `_abort_step`
                # rejects and implicit ones (puzzles.pop KeyError, torch.cat / task_reward / trim_*
                # errors, the empty-batch assert) — broadcasts one poison sentinel before
                # propagating, so workers unblock at Phase C and raise in lockstep.
                try:
                    while puzzles_used < args.examples_per_step:
                        try:
                            msg = result_q.get(timeout=ROLLOUT_GET_TIMEOUT_S)
                        except queue.Empty:
                            if not child.is_alive():
                                raise _abort_step("vLLM child died during generation (no rollouts received)")
                            continue
                        if msg["type"] == MSG_ERROR:
                            raise _abort_step(f"engine child error: {msg['msg']}")
                        pid, version = msg["puzzle_id"], msg["weight_version"]
                        meta = puzzles.pop(pid)
                        enqueue_puzzle(next_puzzle_id); next_puzzle_id += 1
                        if not is_fresh_enough(version, current_version, args.max_staleness):
                            groups_stale += 1
                            consecutive_rejected += 1
                            if consecutive_rejected > reject_limit:
                                raise _abort_step(
                                    f"step {step}: rejected {consecutive_rejected} consecutive rollout groups "
                                    f"(staleness bound too tight or rewards degenerate); aborting")
                            continue
                        rewards, seqs, blps, comps, fins = [], [], [], [], []
                        for s in msg["samples"]:
                            seq = torch.cat([meta["prompt_ids"], torch.tensor(s["token_ids"], dtype=torch.long)])
                            if args.trim_after_answer:
                                seq, lp = trim_generated_with_logprobs(tokenizer, seq, meta["prefix_length"], s["logprobs"])
                            else:
                                lp = s["logprobs"]
                            comp = decode_completion(tokenizer, seq, meta["prefix_length"])
                            rewards.append(task_reward(train_task, meta["conversation"], comp))
                            seqs.append(seq); blps.append(lp); comps.append(comp); fins.append(s.get("finish_reason"))
                        rewards_t = torch.tensor(rewards, dtype=torch.float32)
                        adv = rewards_t - rewards_t.mean()
                        # Unbiased online proxy (before any filter): solved == reward 1.0, matching evaluate().
                        groups_seen_total += 1
                        samples_seen_total += len(rewards)
                        samples_solved_total += sum(1 for r in rewards if r == 1.0)
                        groups_any_solved_total += 1 if any(r == 1.0 for r in rewards) else 0
                        is_zero_var = args.zero_variance_filter and not group_has_signal(rewards)
                        if rollout_fh is not None:
                            status = "zero_variance" if is_zero_var else "trained"
                            for r, a, lp, comp, fin in zip(rewards, adv.tolist(), blps, comps, fins):
                                rollout_fh.write(json.dumps({
                                    "step": step, "weight_version": version,
                                    "staleness": current_version - version,
                                    "puzzle_id": pid, "status": status,
                                    "reward": float(r), "advantage": float(a),
                                    "gen_tokens": len(lp), "finish_reason": fin,
                                    "question": meta["conversation"]["question"], "completion": comp,
                                }, ensure_ascii=False) + "\n")
                        if is_zero_var:
                            groups_zero_variance += 1
                            consecutive_rejected += 1
                            if consecutive_rejected > reject_limit:
                                raise _abort_step(
                                    f"step {step}: rejected {consecutive_rejected} consecutive rollout groups "
                                    f"(staleness bound too tight or rewards degenerate); aborting")
                            continue
                        consecutive_rejected = 0
                        group_solved.append(1.0 if max(rewards) > 0.5 else 0.0)
                        prefill_tokens_total += meta["prefix_length"]
                        for seq, r, a, lp, comp, fin in zip(seqs, rewards, adv.tolist(), blps, comps, fins):
                            step_examples.append(RolloutExample(
                                sequence=seq, prefix_length=meta["prefix_length"], reward=r, advantage=a,
                                behavior_logprobs=lp, weight_version=version, puzzle_id=pid, completion=comp))
                            finish_reasons.append(fin)
                            staleness.append(current_version - version)
                            gen_tokens_total += len(lp)
                        puzzles_used += 1
                    t_collect = time.monotonic() - t_step_start

                    # The collect loop only exits once `examples_per_step` signal groups landed, so
                    # step_examples is non-empty by construction; guard it before scatter so an
                    # empty batch never reaches pad_shards_to_equal -> make_rl_batch_varprefix.
                    assert step_examples, "Phase A produced an empty step batch (would break scatter)"

                    # ---- Phase B (RANK 0 ONLY): batch-normalize advantages over the FULL step batch,
                    # BEFORE scattering. Per-shard std would be wrong, so this must precede the split.
                    adv_all = batch_normalize_advantages(
                        torch.tensor([e.advantage for e in step_examples], dtype=torch.float32, device=device))
                    adv_all_list = adv_all.detach().cpu().tolist()
                    if world_size > 1:
                        pad_example = make_pad_example(pad_token_id)
                        shards = pad_shards_to_equal(step_examples, world_size, pad_example)
                        per_rank = len(shards[0])
                        # Advantages aligned to the padded layout: real examples keep their full-batch
                        # normalized advantage; pad examples carry advantage 0.0 (zero gradient).
                        padded_adv = adv_all_list + [0.0] * (world_size * per_rank - len(adv_all_list))
                        adv_shards = [padded_adv[r * per_rank:(r + 1) * per_rank] for r in range(world_size)]
                    else:
                        shards = [step_examples]
                        adv_shards = [adv_all_list]
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
                payload = _scatter_payload(full_padded, full_adv, abort=False)
                dist.broadcast_object_list(payload, src=0)
                control, full_padded, full_adv = payload
                if control["abort"]:
                    raise RuntimeError("aborting after rank-0 poison sentinel (Phase A error on rank 0)")
                per_rank = len(full_padded) // world_size
                my_shard = full_padded[ddp_rank * per_rank:(ddp_rank + 1) * per_rank]
                my_adv = torch.tensor(
                    full_adv[ddp_rank * per_rank:(ddp_rank + 1) * per_rank],
                    dtype=torch.float32, device=device)
            else:
                my_shard = shards[0]
                my_adv = adv_all

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
            loss_sum = 0.0
            is_ratio_acc = 0.0
            is_clip_acc = 0.0
            is_tok_acc = 0
            num_microbatches = 0
            # ---- Phase E (ALL RANKS): backward over THIS rank's shard with normalizer=1.0 and
            # the GLOBAL counts (identical loss-call args to the single-GPU path). ----
            for start in range(0, total, args.device_batch_size):
                mb = my_shard[start:start + args.device_batch_size]
                seqs = [e.sequence for e in mb]
                prefixes = [e.prefix_length for e in mb]
                input_ids, attention_mask, labels = make_rl_batch_varprefix(seqs, prefixes, pad_token_id, device)
                beh = build_behavior_logprob_tensor_varprefix(seqs, [e.behavior_logprobs for e in mb], prefixes, device)
                mb_stats: dict = {}
                loss = chunked_cispo_loss(
                    model, input_ids, attention_mask, labels, my_adv[start:start + len(mb)],
                    behavior_logprobs=beh, cispo_eps=args.cispo_eps, chunk_size=args.logits_chunk_size,
                    valid_token_normalizer=global_token_count,
                    valid_sample_normalizer=global_sample_count,
                    normalizer=1.0,
                    sequence_normalize=(args.loss_normalization == "sequence"),
                    stats=mb_stats,
                    autocast_dtype=train_autocast_dtype,
                )
                loss.backward()
                loss_sum += float(loss.detach())
                tok = mb_stats.get("valid_tokens", 0)
                is_ratio_acc += mb_stats.get("is_ratio_mean", 0.0) * tok
                is_clip_acc += mb_stats.get("is_clipped_frac", 0.0) * tok
                is_tok_acc += tok
                num_microbatches += 1
            # ---- Phase E (end): manual DP grad reduction — all_reduce(SUM) every param grad so
            # each rank holds the full-batch gradient Σ_r grad(per-shard loss), exactly the
            # single-GPU gradient given normalizer=1.0 + global counts.
            if world_size > 1:
                all_reduce_grads_sum(model)
            # ---- Phase F (ALL RANKS): clip + step on the global gradient; identical grads +
            # identical AdamW state keep every rank's params byte-identical afterward. ----
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            optimizer.step()
            torch.cuda.synchronize()
            t_train = time.monotonic() - t_train_start

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
            t_sync = 0.0
            if master_process:
                try:
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
                    send_thread.join(timeout=WEIGHT_SYNC_TIMEOUT_S)
                    if send_thread.is_alive() or not child.is_alive():
                        raise RuntimeError(
                            "weight broadcast stalled or the vLLM child died during weight sync "
                            f"(child_alive={child.is_alive()})"
                        )
                    if "error" in send_result:
                        raise send_result["error"]
                    _await_status(status_q, MSG_WEIGHTS_READY)
                    t_sync = time.monotonic() - t_sync_start
                except BaseException:
                    if has_next_step:
                        _broadcast_poison(poison_sent)
                    raise
            t_step = time.monotonic() - t_step_start

            # ---- metrics (RANK 0 ONLY): the collect-side counts are rank-0-local; the
            # all-reduced global_token_count drives train throughput. Workers skip to the next
            # step's Phase-C scatter. ----
            num_microbatches_step = num_microbatches
            if master_process:
                try:
                    rewards_all = torch.tensor([e.reward for e in step_examples], dtype=torch.float32)
                    adv_raw = torch.tensor([e.advantage for e in step_examples], dtype=torch.float32)
                    gen_lens = torch.tensor([float(len(e.behavior_logprobs)) for e in step_examples])
                    n_length_trunc = sum(1 for f in finish_reasons if f == "length")
                    n_eos = sum(1 for f in finish_reasons if f == "stop")
                    n_seqs = len(step_examples)
                    metrics = {
                        "step": step, "lr": lr, "weight_version": current_version,
                        "groups/used": puzzles_used,
                        "groups/zero_variance_dropped": groups_zero_variance,
                        "groups/stale_dropped": groups_stale,
                        "seqs": n_seqs,
                        "reward/mean": float(rewards_all.mean()),
                        "reward/std": float(rewards_all.std(unbiased=False)),
                        "reward/solved_frac": float((rewards_all > 0.5).float().mean()),
                        "reward/group_pass_at_k": float(sum(group_solved) / max(1, len(group_solved))),
                        # Unbiased proxies over all fresh generated groups (pre-filter). Use these,
                        # not solved_frac/group_pass_at_k, to gauge live progress; see run_held_out_eval.
                        "reward/online_solved_frac_unfiltered": samples_solved_total / max(1, samples_seen_total),
                        "reward/online_group_any_solved_unfiltered": groups_any_solved_total / max(1, groups_seen_total),
                        "adv/raw_std": float(adv_raw.std(unbiased=False)),
                        "adv/norm_std": float(adv_all.std(unbiased=False)),
                        "loss": loss_sum,
                        "grad_norm": grad_norm,
                        "is_ratio/mean": (is_ratio_acc / is_tok_acc) if is_tok_acc else 0.0,
                        "is_ratio/clipped_frac": (is_clip_acc / is_tok_acc) if is_tok_acc else 0.0,
                        "gen/mean_tokens": float(gen_lens.mean()),
                        "gen/max_tokens": float(gen_lens.max()),
                        "gen/total_tokens": int(gen_tokens_total),
                        "gen/length_trunc_frac": n_length_trunc / max(1, n_seqs),
                        "gen/eos_frac": n_eos / max(1, n_seqs),
                        "staleness/mean": sum(staleness) / max(1, len(staleness)),
                        "staleness/max": (max(staleness) if staleness else 0),
                        "time/collect_s": t_collect,
                        "time/train_s": t_train,
                        "time/sync_s": t_sync,
                        "time/step_s": t_step,
                        "throughput/gen_tokens_per_s": gen_tokens_total / max(1e-6, t_collect),
                        "throughput/train_tokens_per_s": global_token_count / max(1e-6, t_train),
                        "throughput/seqs_per_s": n_seqs / max(1e-6, t_step),
                        "throughput/result_q_backlog": _safe_qsize(result_q),
                    }
                    metrics.update(estimate_step_perf_metrics(
                        model_info=model_perf_info,
                        train_padded_tokens=float(global_token_count),
                        train_forward_backward_passes=float(num_microbatches_step),
                        rollout_prefill_tokens=float(prefill_tokens_total),
                        rollout_decode_tokens=float(gen_tokens_total),
                        rollout_forward_passes=float(gen_tokens_total),
                        train_seconds=t_train,
                        rollout_seconds=t_collect,
                    ))

                    print0(
                        f"step {step:3d} | rew {metrics['reward/mean']:.3f} pass@k {metrics['reward/group_pass_at_k']:.2f} "
                        f"solved {metrics['reward/solved_frac']:.2f} | loss {loss_sum:.4f} gnorm {grad_norm:.2f} "
                        f"| IS {metrics['is_ratio/mean']:.2f} clip {metrics['is_ratio/clipped_frac']:.2f} "
                        f"| genlen {metrics['gen/mean_tokens']:.0f} trunc {metrics['gen/length_trunc_frac']:.2f} "
                        f"| gen {metrics['throughput/gen_tokens_per_s']:.0f} tok/s tr {metrics['throughput/train_tokens_per_s']:.0f} tok/s "
                        f"| t {t_step:.1f}s (gen {t_collect:.1f}/tr {t_train:.1f}/sync {t_sync:.1f}) "
                        f"| v{current_version} stale {metrics['staleness/mean']:.1f}"
                    )
                    wandb_run.log(metrics)
                    if rollout_fh is not None:
                        rollout_fh.flush()
                    if wandb_rollouts_enabled and n_seqs > 0:
                        # Sample a spread of this step's rollouts by reward (so the table shows
                        # solved + unsolved + middle), append to the running W&B table.
                        order = sorted(range(n_seqs), key=lambda i: step_examples[i].reward)
                        n = min(args.wandb_rollout_samples, n_seqs)
                        picks = [order[round(j * (n_seqs - 1) / (n - 1))] for j in range(n)] if n > 1 else [order[0]]
                        for i in picks:
                            e = step_examples[i]
                            wandb_rollout_rows.append([
                                step, e.weight_version, staleness[i], e.puzzle_id, "trained",
                                float(e.reward), float(e.advantage), len(e.behavior_logprobs),
                                finish_reasons[i], e.completion[:4000],
                            ])
                        wandb_rollout_rows = wandb_rollout_rows[-400:]  # cap table size
                    if should_save_checkpoint_for_step(step, num_steps, args.save_every, args.save_final):
                        checkpoint_dir = save_hf_checkpoint(model, tokenizer, run_dir, step)
                        print0(f"Saved checkpoint to {checkpoint_dir}")
                except BaseException:
                    if has_next_step:
                        _broadcast_poison(poison_sent)
                    raise
    finally:
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
        wandb_run.finish()
        if rollout_fh is not None:
            rollout_fh.close()
        if world_size > 1:
            compute_cleanup()


def get_lr_multiplier(
    step: int,
    num_steps: int,
    init_lr_frac: float = 1.0,
    warmup_steps: int = 0,
    schedule: str = "linear",
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
    if warmup_steps == 0:
        if num_steps <= 1:
            return 1.0
        if step == 0:
            return init_lr_frac
        return max(0.0, 1.0 - (step - 1) / num_steps)
    decay_steps = max(1, num_steps - warmup_steps)
    decay_progress = (step - warmup_steps) / decay_steps
    return max(0.0, 1.0 - decay_progress)


def save_hf_checkpoint(
    model,
    tokenizer,
    output_dir: Path,
    step: int,
) -> Path:
    checkpoint_dir = output_dir / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir


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
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **loader_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **loader_kwargs)
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if dtype != torch.float32 and hasattr(model, "lm_head") and isinstance(model.lm_head, torch.nn.Linear):
        model.lm_head = FP32LMHead(model.lm_head)
        model.lm_head.to(device)
        if getattr(model.config, "tie_word_embeddings", False):
            model.config.tie_word_embeddings = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    return model, tokenizer


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
                             "too small => 100%% length-truncation => 0 reward => no training signal.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling; 0 disables")
    parser.add_argument("--top-p", type=float, default=0.7, help="Nucleus sampling threshold; 0 disables")
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
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--loss-normalization",
        choices=["token", "sequence"],
        default="sequence",
        help="Normalize policy-gradient loss by rollout tokens or by each sequence before averaging.",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                        help="Trade compute for activation memory. On by default: the fp32 trainer GPU "
                             "(params+Adam+grad ~64GB) plus 6144-token rollouts otherwise OOMs.")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prompt-style",
        choices=["nanochat", "sokoban", "brief", "reason", "instruct", "rg"],
        default="rg",
        help="nanochat uses the raw puzzle; rg matches the reasoning_gym task framing/format; sokoban/brief/reason/instruct require reasoning before the final-answer marker",
    )
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
    parser.add_argument("--inflight-requests", type=int, default=16,
                        help="Target number of puzzle generations kept in flight")
    parser.add_argument("--max-staleness", type=int, default=4,
                        help="Drop rollouts older than this many weight versions (PipelineRL-k)")
    parser.add_argument("--logits-chunk-size", type=int, default=1024,
                        help="Sequence chunk size for the memory-efficient chunked LM-head CISPO loss")
    parser.add_argument("--train-autocast-dtype", choices=["none", "bfloat16"], default="bfloat16",
                        help="Autocast dtype for the trainer's transformer-body forward; the LM head "
                             "stays fp32. 'none' runs the body in fp32 (more activation memory). bf16 "
                             "halves activation memory and matches the bf16 vLLM generator.")
    parser.add_argument("--debug-assert-replica-sync", action="store_true",
                        help="After each optimizer step (first few steps), assert all trainer-rank "
                             "model replicas are bit-identical via an all-reduce MIN==MAX checksum. "
                             "Debug guard; small overhead. No-op at world_size==1.")
    parser.add_argument("--cispo-eps", type=float, default=4.0,
                        help="CISPO upper clip epsilon_max for the importance weight")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="vLLM max_model_len (prompt + generation) for the generators; must cover "
                             "prompt (~700) + --max-new-tokens (6144).")
    parser.add_argument("--zero-variance-filter", action=argparse.BooleanOptionalAction, default=True,
                        help="Drop puzzle groups whose rewards are all equal (zero advantage). "
                             "Disable to keep them (e.g. plumbing smoke tests where the model gets no reward).")
    parser.add_argument("--save-rollouts", action=argparse.BooleanOptionalAction, default=True,
                        help="Write every decoded rollout (prompt, completion, reward, advantage, finish "
                             "reason, staleness, disposition) to <output-dir>/<run>/rollouts.jsonl for later analysis.")
    parser.add_argument("--wandb-rollout-samples", type=int, default=8,
                        help="Rollouts sampled per step (spread by reward) into a browsable W&B Table; 0 disables.")

    # Standalone held-out evaluation (the authoritative leaderboard metric). Runs in its own
    # process with a dedicated vLLM engine sized for the full rollout budget; no torchrun/DDP.
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate a checkpoint (or the base model) on the held-out set and exit.")
    parser.add_argument("--eval-checkpoint", type=Path, default=None,
                        help="Checkpoint dir to evaluate; defaults to --model (evaluate the base model).")
    parser.add_argument("--eval-k", type=int, default=16,
                        help="Samples per puzzle for --eval-only; k=16 enables pass@{1,4,8,16}.")
    parser.add_argument("--eval-max-tokens", type=int, default=6144,
                        help="Generation budget per puzzle for --eval-only (leaderboard protocol).")
    parser.add_argument("--eval-max-model-len", type=int, default=8192,
                        help="vLLM max_model_len for the eval engine; must be >= prompt + --eval-max-tokens.")
    parser.add_argument("--eval-temperature", type=float, default=0.8, help="--eval-only sampling temperature.")
    parser.add_argument("--eval-top-p", type=float, default=0.7, help="--eval-only nucleus sampling threshold.")
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
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Evaluate only the first N puzzles (smoke tests); default = full eval set.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

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

    if args.device_batch_size < 1:
        raise ValueError("--device-batch-size must be at least 1")
    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2 so per-puzzle advantages are non-zero")
    if not (0.0 < args.init_lr_frac <= 1.0):
        raise ValueError("--init-lr-frac must be in (0, 1]")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.adam_eps <= 0:
        raise ValueError("--adam-eps must be positive")
    if not (0.0 <= args.top_p <= 1.0):
        raise ValueError("--top-p must be in [0, 1]")

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
