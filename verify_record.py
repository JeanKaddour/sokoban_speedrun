"""Offline verifier for leaderboard record submissions.

Usage:
    python verify_record.py records/<dir> [--eval-data datasets/sokoban_eval.jsonl]
                            [--target 0.50] [--alpha 0.01] [--allow-missing-rollouts]

For every eval_*.json in the record directory this re-derives the submission's claims
without a GPU, importing the extractor/scorer FROM speedrun.py so verification cannot
drift from what training and eval actually scored:

 1. AGGREGATES  — pass@1 / pass@k / answer_rate / CI re-derived from the JSON's own
    per-puzzle arrays (always possible, even for records predating rollout artifacts).
 2. RE-SCORE    — every completion in <eval_json_stem>.rollouts.jsonl.gz re-extracted and
    re-scored with ReasoningGym; per-puzzle solved counts must match the JSON exactly.
 3. DATASET     — the artifact's eval_data_sha256 must match the committed eval set.
 4. HEALTH      — degenerate-tail fraction (zlib-compressible tails => repetition loops),
    duplicate-completion rate (sampler collapse), finish-reason mix, answered-rate match.

With >=2 eval JSONs it then re-runs the leaderboard significance verdict (same one-sided
t-test as speedrun.run_standalone_eval). Exit code: 0 PASS, 1 FAIL.

Heavy imports note: speedrun.py imports torch/transformers at module level; run via
    uv run --with torch --with transformers --with reasoning-gym python verify_record.py ...
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import statistics
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speedrun import (  # noqa: E402  (single source of truth for scoring)
    _bootstrap_ci,
    _file_sha256,
    _pass_at_k_unbiased,
    _wilson_ci,
    extract_sokoban_answer,
    load_sokoban_jsonl_dataset,
    student_t_sf,
)

DEGENERATE_TAIL_RATIO = 0.15   # zlib(tail)/len(tail) below this == repetition loop
DEGENERATE_MAX_FRAC = 0.01     # more than 1% degenerate tails fails the health check
DUP_WARN_FRAC = 0.10           # >10% duplicate completions within puzzles: warn only
FLOAT_TOL = 1e-9


class Failures:
    def __init__(self) -> None:
        self.items: list[str] = []

    def check(self, ok: bool, msg: str) -> bool:
        if not ok:
            self.items.append(msg)
            print(f"  FAIL  {msg}")
        return ok


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= FLOAT_TOL


def _tail_ratio(text: str, n: int = 4000) -> float:
    tail = text[-n:].encode()
    return len(zlib.compress(tail, 6)) / max(1, len(tail))


def verify_aggregates(record: dict, fails: Failures, tag: str) -> None:
    """Check 1: the JSON's headline numbers must follow from its own per-puzzle arrays."""
    n = record["n_puzzles"]
    fracs = record["per_puzzle_solve_frac"]
    ns = record["per_puzzle_n"]
    cs = record["per_puzzle_solved_count"]
    fails.check(len(fracs) == len(ns) == len(cs) == n, f"{tag}: per-puzzle array lengths != n_puzzles")
    fails.check(all(_close(f, c / max(1, nn)) for f, c, nn in zip(fracs, cs, ns)),
                f"{tag}: per_puzzle_solve_frac inconsistent with solved/n")
    fails.check(_close(record["pass_at_1"], sum(fracs) / max(1, n)),
                f"{tag}: pass_at_1 != mean(per-puzzle fracs)")
    for j_str, claimed in record["pass_at_k"].items():
        j = int(j_str)
        derived = sum(_pass_at_k_unbiased(nn, c, min(j, nn)) for nn, c in zip(ns, cs)) / max(1, n)
        fails.check(_close(claimed, derived), f"{tag}: pass_at_k[{j}] {claimed} != derived {derived}")
    fails.check(_close(record["answer_rate"], sum(record["per_puzzle_answered_count"]) / max(1, sum(ns))),
                f"{tag}: answer_rate inconsistent with per-puzzle answered counts")
    seed = int(record["sampling"].get("seed") or 0)
    if record["k"] == 1:
        lo, hi = _wilson_ci(sum(cs), sum(ns))
    else:
        lo, hi = _bootstrap_ci(fracs, seed=seed)
    fails.check(_close(record["ci_low"], lo) and _close(record["ci_high"], hi),
                f"{tag}: CI [{record['ci_low']}, {record['ci_high']}] != re-derived [{lo}, {hi}]")


def verify_rollouts(record: dict, rollouts_path: Path, eval_task, eval_data_path: Path,
                    fails: Failures, tag: str) -> None:
    """Checks 2-4: re-score the saved completions and run health checks."""
    with gzip.open(rollouts_path, "rt", encoding="utf-8") as fh:
        meta = json.loads(fh.readline())
        rows = [json.loads(line) for line in fh]
    fails.check(meta.get("type") == "meta", f"{tag}: rollouts artifact missing meta header")

    local_sha = _file_sha256(eval_data_path)
    fails.check(meta.get("eval_data_sha256") == local_sha,
                f"{tag}: eval_data sha256 mismatch (artifact {meta.get('eval_data_sha256')!r:.20} "
                f"vs local {local_sha[:12]}...) — wrong or modified eval set")

    n, k = record["n_puzzles"], record["k"]
    fails.check(len(rows) == sum(record["per_puzzle_n"]),
                f"{tag}: artifact has {len(rows)} rows, JSON claims {sum(record['per_puzzle_n'])} samples")

    resolved: dict[int, int] = {}
    reanswered: dict[int, int] = {}
    n_degenerate = 0
    n_dup = 0
    finish: dict[str, int] = {}
    by_puzzle_texts: dict[int, set[str]] = {}
    for row in rows:
        idx = row["puzzle_idx"]
        completion = row["completion"]
        moves = extract_sokoban_answer(completion)
        fails.check(moves == row["moves"],
                    f"{tag}: puzzle {idx} sample {row['sample']}: extracted moves {moves!r} "
                    f"!= recorded {row['moves']!r}")
        solved = eval_task.evaluate(eval_task[idx], completion)
        fails.check(solved == row["solved"],
                    f"{tag}: puzzle {idx} sample {row['sample']}: re-scored solved={solved} "
                    f"!= recorded {row['solved']}")
        resolved[idx] = resolved.get(idx, 0) + solved
        reanswered[idx] = reanswered.get(idx, 0) + (moves is not None)
        finish[row["finish_reason"]] = finish.get(row["finish_reason"], 0) + 1
        if _tail_ratio(completion) < DEGENERATE_TAIL_RATIO:
            n_degenerate += 1
        seen = by_puzzle_texts.setdefault(idx, set())
        if completion in seen:
            n_dup += 1
        seen.add(completion)

    for pos in range(n):
        fails.check(resolved.get(pos, 0) == record["per_puzzle_solved_count"][pos],
                    f"{tag}: puzzle {pos}: re-scored solved count {resolved.get(pos, 0)} "
                    f"!= JSON {record['per_puzzle_solved_count'][pos]}")
        fails.check(reanswered.get(pos, 0) == record["per_puzzle_answered_count"][pos],
                    f"{tag}: puzzle {pos}: re-extracted answered count {reanswered.get(pos, 0)} "
                    f"!= JSON {record['per_puzzle_answered_count'][pos]}")

    deg_frac = n_degenerate / max(1, len(rows))
    fails.check(deg_frac <= DEGENERATE_MAX_FRAC,
                f"{tag}: {deg_frac:.1%} of completions have degenerate (repetitive) tails "
                f"(limit {DEGENERATE_MAX_FRAC:.0%})")
    dup_frac = n_dup / max(1, len(rows))
    if dup_frac > DUP_WARN_FRAC:
        print(f"  WARN  {tag}: {dup_frac:.1%} duplicate completions within puzzles (sampler collapse?)")
    mix = ", ".join(f"{r}={c}" for r, c in sorted(finish.items()))
    print(f"  ok    {tag}: re-scored {len(rows)} completions (k={k}); finish reasons: {mix}; "
          f"degenerate tails {deg_frac:.2%}, dups {dup_frac:.2%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record_dir", type=Path, help="records/<date>_<nn>_<name>/ directory to verify")
    ap.add_argument("--eval-data", type=Path, default=Path("datasets/sokoban_eval.jsonl"))
    ap.add_argument("--target", type=float, default=0.50, help="leaderboard pass@1 bar")
    ap.add_argument("--alpha", type=float, default=0.01, help="one-sided significance level")
    ap.add_argument("--allow-missing-rollouts", action="store_true",
                    help="Verify JSON-level claims only for records predating rollout artifacts.")
    args = ap.parse_args()

    eval_jsons = sorted(glob.glob(str(args.record_dir / "eval_*.json")))
    if not eval_jsons:
        print(f"no eval_*.json found in {args.record_dir}")
        return 1
    eval_task = load_sokoban_jsonl_dataset(args.eval_data, split_name="eval")

    fails = Failures()
    records = []
    for path in eval_jsons:
        record = json.loads(Path(path).read_text())
        tag = Path(path).name
        records.append(record)
        print(f"== {tag} (seed {record.get('seed')}, ckpt {record.get('checkpoint')}) ==")
        verify_aggregates(record, fails, tag)
        rollouts_name = record.get("rollouts_file") or (Path(path).stem + ".rollouts.jsonl.gz")
        rollouts_path = Path(path).with_name(rollouts_name)
        if rollouts_path.exists():
            verify_rollouts(record, rollouts_path, eval_task, args.eval_data, fails, tag)
        else:
            msg = f"{tag}: no rollouts artifact at {rollouts_path.name} — completions not re-scored"
            if args.allow_missing_rollouts:
                print(f"  WARN  {msg}")
            else:
                fails.check(False, msg)

    # Multi-seed significance verdict — same math as speedrun.run_standalone_eval.
    values = [float(r["pass_at_1"]) for r in records]
    K = len(values)
    print(f"\n=== Significance: mean(pass@1) > {args.target} over {K} seed(s) ===")
    if K >= 2:
        mean, sd = statistics.mean(values), statistics.stdev(values)
        se = sd / math.sqrt(K)
        t = ((mean - args.target) / se if se > 0 else
             (math.inf if mean > args.target else -math.inf))
        p = student_t_sf(t, K - 1) if se > 0 else (0.0 if mean > args.target else 1.0)
        significant = p < args.alpha and mean > args.target
        print(f"  values {[round(v, 4) for v in values]} | mean {mean:.4f} +/- {sd:.4f} | "
              f"t={t:.3f} p={p:.5f} (alpha={args.alpha})")
        fails.check(significant, f"significance test failed (mean {mean:.4f}, p={p:.5f})")
    else:
        print(f"  single seed: pass@1 {values[0]:.4f}, ci_low {records[0]['ci_low']:.4f} "
              f"(target {args.target}) — no t-test possible")
        fails.check(records[0]["ci_low"] > args.target,
                    f"single-seed ci_low {records[0]['ci_low']:.4f} does not clear {args.target}")

    print(f"\n{'VERDICT: PASS' if not fails.items else f'VERDICT: FAIL ({len(fails.items)} problem(s))'}")
    return 0 if not fails.items else 1


if __name__ == "__main__":
    raise SystemExit(main())
