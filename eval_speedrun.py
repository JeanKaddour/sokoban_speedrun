"""Standalone held-out eval for Sokoban Speedrun.

Public entrypoint:
    python -m eval_speedrun --eval-checkpoint <checkpoint-or-model>
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

from speedrun import (
    INTERRUPTION_TEXT,
    INTERRUPTION_TEXT_NO_THINK,
    _add_vllm_tuning_args,
    _git_commit,
    build_async_engine,
    default_run_name,
    encode_prompt,
    extract_sokoban_answer,
    generate_group,
    load_sokoban_jsonl_dataset,
    make_interrupt_config,
    sanitize_run_name,
    vllm_tuning_from_args,
)


def _file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if k > n:
        raise ValueError(f"pass@k requires k<=n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-puzzle solve fractions."""
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
    """Evaluate one policy on the fixed held-out task."""
    if indices is None:
        indices = list(range(len(eval_task)))
    eval_sampling = dict(sampling)
    eval_sampling["logprobs"] = 0
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
        for j, sample in enumerate(samples):
            completion = tokenizer.decode(sample["token_ids"], skip_special_tokens=True)
            moves = extract_sokoban_answer(completion)
            if moves is None:
                extract_fail += 1
            else:
                answered += 1
            if sample.get("finish_reason") == "length":
                length_trunc += 1
            solved = eval_task.evaluate(conv, completion)
            c += solved
            if collect_rollouts:
                rows.append({
                    "puzzle_idx": idx,
                    "sample": j,
                    "solved": solved,
                    "moves": moves,
                    "finish_reason": sample.get("finish_reason"),
                    "gen_tokens": len(sample["token_ids"]),
                    "completion": completion,
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
    pass_at_1 = sum(per_puzzle) / max(1, n_puzzles)

    n_short = sum(1 for r in results if r["n"] < k)
    if n_short:
        print(
            f"WARNING: {n_short}/{n_puzzles} puzzles returned fewer than k={k} samples; "
            "pass@k clamps k to each puzzle's sample count",
            flush=True,
        )
    pass_at_k = {
        j: sum(_pass_at_k_unbiased(r["n"], r["c"], min(j, r["n"])) for r in results) / max(1, n_puzzles)
        for j in pass_at_ks
        if j <= k
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


def _eval_one_checkpoint(args: argparse.Namespace, model_path: str, multi: bool) -> dict:
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
        min_p=args.eval_min_p,
        max_tokens=args.eval_max_tokens,
        seed=args.eval_seed,
        logprobs=0,
    )
    if args.eval_interruption:
        marker_ids = tokenizer(
            INTERRUPTION_TEXT if args.enable_thinking else INTERRUPTION_TEXT_NO_THINK,
            add_special_tokens=False,
        )["input_ids"]
        sampling["interrupt"] = make_interrupt_config(
            marker_ids,
            args.eval_interrupt_answer_tokens,
            args.eval_max_model_len,
            base_temperature=args.eval_temperature,
            base_top_p=args.eval_top_p,
            base_top_k=args.eval_top_k,
            base_min_p=args.eval_min_p,
            temperature=args.eval_interrupt_temperature,
            top_p=args.eval_interrupt_top_p,
            top_k=args.eval_interrupt_top_k,
            min_p=args.eval_interrupt_min_p,
        )

    print(
        f"[eval] model={model_path} eval_data={args.eval_data} "
        f"n={len(indices) if indices is not None else len(eval_task)} k={args.eval_k} "
        f"max_tokens={args.eval_max_tokens} sampling=temp{args.eval_temperature}/top_p{args.eval_top_p}/"
        f"top_k{args.eval_top_k}/min_p{args.eval_min_p}/seed{args.eval_seed} "
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
                engine,
                tokenizer,
                eval_task,
                k=args.eval_k,
                sampling=sampling,
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
            "n_puzzles",
            "k",
            "pass_at_1",
            "pass_at_k",
            "ci_low",
            "ci_high",
            "se",
            "n_extract_fail",
            "n_answered",
            "n_length_trunc",
            "answer_rate",
            "solve_given_answer",
            "trunc_frac",
            "sampling",
            "per_puzzle_solve_frac",
            "per_puzzle_n",
            "per_puzzle_solved_count",
            "per_puzzle_answered_count",
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
            ckpt_tag = sanitize_run_name(Path(model_path).parent.name or model_path)
            out_path = args.output_dir / safe / f"eval_{ckpt_tag}_{suffix}.json"
        else:
            out_path = args.output_dir / safe / f"eval_{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rollouts = result.get("rollouts")
    if rollouts is not None:
        rollouts_path = out_path.with_name(out_path.stem + ".rollouts.jsonl.gz")
        meta = {
            "type": "meta",
            "seed": args.eval_seed,
            "run": args.run,
            "step": step,
            "checkpoint": model_path,
            "model": args.model,
            "eval_data": str(args.eval_data),
            "eval_data_sha256": _file_sha256(args.eval_data),
            "k": args.eval_k,
            "n_puzzles": result["n_puzzles"],
            "git_commit": _git_commit(),
            "sampling": result["sampling"],
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


def run_eval(args: argparse.Namespace) -> None:
    checkpoints = [str(c) for c in args.eval_checkpoint] if args.eval_checkpoint else [args.model]
    if args.eval_output is not None and len(checkpoints) > 1:
        sys.exit("error: --eval-output is only valid with a single checkpoint; multi-checkpoint evals auto-name one JSON per checkpoint.")
    for model_path in checkpoints:
        _eval_one_checkpoint(args, model_path, multi=len(checkpoints) > 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the fixed Sokoban held-out set",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders")
    parser.add_argument("--eval-data", type=Path, default=Path("datasets/sokoban_eval.jsonl"))
    parser.add_argument("--run", type=str, default=None, help="Run/output name. Defaults to sokoban-eval-<date>.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Base directory for eval JSONs")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-save-rollouts", action=argparse.BooleanOptionalAction, default=True,
                        help="Write every eval completion to <eval-json-stem>.rollouts.jsonl.gz.")
    parser.add_argument("--eval-checkpoint", type=Path, nargs="+", default=None,
                        help="Checkpoint dir(s) to evaluate sequentially; defaults to --model.")
    parser.add_argument("--eval-k", type=int, default=8, help="Samples per puzzle.")
    parser.add_argument("--eval-max-tokens", type=int, default=12288, help="Generation budget per puzzle.")
    parser.add_argument("--eval-max-model-len", type=int, default=16384,
                        help="vLLM max_model_len for the eval engine.")
    parser.add_argument("--eval-interruption", action=argparse.BooleanOptionalAction, default=True,
                        help="Use interruption-based answer forcing.")
    parser.add_argument("--eval-interrupt-answer-tokens", type=int, default=512,
                        help="Token budget for the forced final answer.")
    parser.add_argument("--eval-interrupt-temperature", type=float, default=None,
                        help="Sampling temperature for forced final-answer continuations; default inherits --eval-temperature.")
    parser.add_argument("--eval-interrupt-top-p", type=float, default=None,
                        help="Nucleus sampling threshold for forced final-answer continuations; default inherits --eval-top-p.")
    parser.add_argument("--eval-interrupt-top-k", type=int, default=None,
                        help="Top-k sampling for forced final-answer continuations; default inherits --eval-top-k.")
    parser.add_argument("--eval-interrupt-min-p", type=float, default=None,
                        help="Min-p sampling for forced final-answer continuations; default inherits --eval-min-p.")
    parser.add_argument("--eval-temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--eval-top-p", type=float, default=0.95, help="Nucleus sampling threshold.")
    parser.add_argument("--eval-top-k", type=int, default=0, help="Top-k sampling; 0 disables.")
    parser.add_argument("--eval-min-p", type=float, default=0.0,
                        help="Min-p sampling; 0 disables.")
    parser.add_argument("--eval-seed", type=int, default=12345,
                        help="Sampling/bootstrap seed.")
    parser.add_argument("--eval-output", type=Path, default=None,
                        help="Per-run eval JSON path; default <output-dir>/<run>/eval_<suffix>.json.")
    parser.add_argument("--eval-step", type=int, default=None,
                        help="Step number recorded in the eval JSON; parsed from checkpoint path if omitted.")
    parser.add_argument("--eval-vllm-dp", type=int, default=1, help="Data-parallel GPUs for the eval engine.")
    parser.add_argument("--eval-gpu-mem-util", type=float, default=0.9,
                        help="gpu_memory_utilization for the eval engine.")
    parser.add_argument("--eval-vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=False,
                        help="Pass enforce_eager=True to the eval vLLM engine.")
    _add_vllm_tuning_args(
        parser,
        prefix="eval_",
        ctx="the standalone eval engine",
        seqs_note="Default is 32 for the long-context record protocol. Raise with --eval-concurrency.",
        batched_note="Default is 40960 for the long-context record protocol.",
    )
    parser.set_defaults(eval_vllm_max_num_batched_tokens=40960, eval_vllm_max_num_seqs=32)
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Evaluate only the first N puzzles; default = full eval set.")
    parser.add_argument("--eval-concurrency", type=int, default=32,
                        help="Max concurrent in-flight eval requests.")
    return parser


_VLLM_POSITIVE_KEYS = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_num_scheduled_tokens",
    "max_num_partial_prefills",
    "max_long_partial_prefills",
    "stream_interval",
)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("eval_temperature", "eval_interrupt_temperature"):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("eval_top_p", "eval_interrupt_top_p", "eval_min_p", "eval_interrupt_min_p"):
        value = getattr(args, name)
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    for name in ("eval_top_k", "eval_interrupt_top_k"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for attr, minimum, requirement in (
        ("eval_k", 1, "at least 1"),
        ("eval_max_tokens", 1, "positive"),
        ("eval_max_model_len", 1, "positive"),
        ("eval_interrupt_answer_tokens", 1, "positive"),
        ("eval_concurrency", 1, "at least 1"),
        ("eval_vllm_dp", 1, "at least 1"),
    ):
        if getattr(args, attr) < minimum:
            raise ValueError(f"--{attr.replace('_', '-')} must be {requirement}")
    if args.eval_limit is not None and args.eval_limit < 0:
        raise ValueError("--eval-limit must be non-negative")
    for key in _VLLM_POSITIVE_KEYS:
        value = getattr(args, f"eval_vllm_{key}")
        if value is not None and value < 1:
            raise ValueError(f"--eval-vllm-{key.replace('_', '-')} must be positive")
    threshold = args.eval_vllm_long_prefill_token_threshold
    if threshold is not None and threshold < 0:
        raise ValueError("--eval-vllm-long-prefill-token-threshold must be non-negative")
    partial = args.eval_vllm_max_num_partial_prefills
    long_partial = args.eval_vllm_max_long_partial_prefills
    if (long_partial if long_partial is not None else 1) > (partial if partial is not None else 1):
        raise ValueError("--eval-vllm-max-long-partial-prefills must be <= --eval-vllm-max-num-partial-prefills")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if not args.run:
        args.run = default_run_name("eval")
    _validate_args(args)
    run_eval(args)


if __name__ == "__main__":
    main()
