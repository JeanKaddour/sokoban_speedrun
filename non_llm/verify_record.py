"""Offline verifier for non-LLM (PufferLib boxoban) leaderboard record submissions.

Usage:
    cd non_llm && uv run python verify_record.py records/<dir>
    (default --target comes from the track gate, 0.70)

For every primary eval (eval_seed<TRAINSEED>.json) in the record directory — and the optional
verification/ rerun — this re-derives the record's claims WITHOUT a GPU, importing the aggregate
helpers FROM speedrun_non_llm so verification can never drift from what evaluate() actually computed:

 1. AGGREGATES — pass@1 / pass@k / answer_rate / CI re-derived from the JSON's own per-puzzle arrays.
 2. HELD-OUT  — the committed eval bin's sha256 must match the record, and every scored level
    (per_puzzle_level_sha) must be a level in that bin. This pins the eval to the canonical
    DeepMind test split and is the offline disjointness/integrity check (training draws the
    disjoint official train split). Records predating these fields are aggregate-verified only.
 3. SOURCE    — each source/speedrun_<seed>.py snapshot must match the embedded run-log source.

It then applies the leaderboard gate: each submission run, and each verification rerun when present,
must have lower 95% CI > target. Exit code: 0 PASS, 1 FAIL.

Heavy imports note: speedrun_non_llm imports torch at module level (no CUDA on import), so run via uv.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))   # repo root, for make_record_report (FLOPs formatting)
from speedrun_non_llm import (  # noqa: E402  (single source of truth for eval aggregates)
    GRID,
    OBS_CHANNELS,
    _bootstrap_ci,
    _file_sha256,
    _pass_at_k_unbiased,
    _wilson_ci,
)

FLOAT_TOL = 1e-9
SOURCE_DIVIDER = "=" * 100
PUZZLE_BYTES = OBS_CHANNELS * GRID * GRID + 5   # 405: [agent,walls,boxes,targets](400) + meta(5)
BOARD_BYTES = OBS_CHANNELS * GRID * GRID        # 400: the obs board — uniquely identifies a level


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


def bin_level_shas(path: Path) -> set[str]:
    """sha256 of every level board (first 400 bytes of each 405-byte puzzle) in a boxoban bin —
    the same board-identity hash evaluate() records per scored level."""
    data = path.read_bytes()
    return {hashlib.sha256(data[i * PUZZLE_BYTES: i * PUZZLE_BYTES + BOARD_BYTES]).hexdigest()
            for i in range(len(data) // PUZZLE_BYTES)}


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


def verify_holdout(record: dict, fails: Failures, tag: str, data_dir: Path) -> None:
    """Check 2: the eval ran on the committed canonical test bin (offline disjointness/integrity).

    Records predating these fields are aggregate-verified only (WARN, not FAIL) — current evaluate()
    always emits them, so a missing field on a fresh submission is itself a visible red flag."""
    shas = record.get("per_puzzle_level_sha")
    bin_name = record.get("holdout_bin")
    if not shas or not bin_name:
        print(f"  WARN  {tag}: record predates held-out pinning (no per_puzzle_level_sha/holdout_bin)")
        return
    bin_path = data_dir / bin_name
    if not bin_path.exists():
        fails.check(False, f"{tag}: committed eval bin {bin_path} not found")
        return
    local_sha = _file_sha256(bin_path)
    fails.check(local_sha == record.get("holdout_bin_sha256"),
                f"{tag}: eval bin sha256 mismatch (local {str(local_sha)[:12]} vs "
                f"record {str(record.get('holdout_bin_sha256'))[:12]}) — wrong or modified eval set")
    pool = bin_level_shas(bin_path)
    expected = record.get("holdout_n_levels")
    fails.check(expected is None or len(pool) == expected,
                f"{tag}: eval bin has {len(pool)} levels, record claims {expected}")
    missing = [s for s in shas if s not in pool]
    fails.check(not missing, f"{tag}: {len(missing)} scored level(s) are NOT in the committed eval bin "
                             f"— eval pool does not match the leaderboard test split")
    fails.check(len(shas) == record["n_puzzles"] and len(set(shas)) == len(shas),
                f"{tag}: per_puzzle_level_sha length/uniqueness inconsistent with n_puzzles")
    if not missing:
        print(f"  ok    {tag}: eval bin {bin_name} sha {str(local_sha)[:12]}; "
              f"all {len(shas)} scored levels in pool (coverage {len(shas)}/{len(pool)})")


def verify_source_snapshots(record_dir: Path, fails: Failures, label: str) -> None:
    for log_path in sorted(record_dir.glob("train_log_seed*.txt")):
        if log_path.name.endswith(".flops.txt"):
            continue
        seed = log_path.stem.replace("train_log_", "")
        snapshot = record_dir / "source" / f"speedrun_{seed}.py"
        tag = f"{label} {seed}"
        if not snapshot.exists():
            fails.check(False, f"{tag}: missing source snapshot {snapshot.name}")
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        embedded, sep, _rest = text.partition(f"\n{SOURCE_DIVIDER}\n")
        if not sep:
            fails.check(False, f"{tag}: could not find run-log source divider")
            continue
        embedded = embedded if embedded.endswith("\n") else embedded + "\n"
        saved = snapshot.read_text(encoding="utf-8")
        fails.check(saved == embedded, f"{tag}: source snapshot does not match embedded run-log source")
        if saved == embedded:
            sha = hashlib.sha256(saved.encode("utf-8")).hexdigest()
            print(f"  ok    {tag}: source snapshot source/{snapshot.name} sha256 {sha[:12]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record_dir", type=Path, help="records/<date>_<nn>_<name>/ directory to verify")
    ap.add_argument("--target", type=float, default=0.70, help="leaderboard solve-rate bar")
    ap.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data",
                    help="directory holding the committed boxoban_<diff>_<split>.bin eval bins")
    args = ap.parse_args()

    primary_jsons = sorted(glob.glob(str(args.record_dir / "eval_seed*.json")))
    if not primary_jsons:
        print(f"no eval_seed*.json found in {args.record_dir}")
        return 1

    fails = Failures()

    def seed_label(path: str) -> str:
        return Path(path).stem.replace("eval_", "").split(".")[0]

    def check(path: str, tag: str, data_dir: Path) -> dict:
        record = json.loads(Path(path).read_text())
        print(f"== {tag} (seed {record.get('seed')}, ckpt {record.get('checkpoint')}) ==")
        verify_aggregates(record, fails, tag)
        verify_holdout(record, fails, tag, data_dir)
        return record

    submission = [(seed_label(p), check(p, Path(p).name, args.data_dir)) for p in primary_jsons]
    verify_source_snapshots(args.record_dir, fails, "submission")

    verif_jsons = sorted(glob.glob(str(args.record_dir / "verification" / "eval_seed*.json")))
    verification = [(seed_label(p), check(p, f"verification/{Path(p).name}", args.data_dir)) for p in verif_jsons]
    if verification:
        verify_source_snapshots(args.record_dir / "verification", fails, "verification")

    # Gate: every run — submission AND verification — independently clears the target via its
    # lower 95% CI (single-seed protocol; no cross-seed averaging).
    print(f"\n=== Gate: each run's lower 95% CI > {args.target} ===")

    def gate(label: str, runs: list) -> None:
        for sl, r in runs:
            ok = r["ci_low"] > args.target
            print(f"  {label} {sl}: pass@1 {r['pass_at_1']:.4f}, ci_low {r['ci_low']:.4f} "
                  f"(target {args.target}) — {'CLEARS' if ok else 'DOES NOT CLEAR'}")
            fails.check(ok, f"{label} {sl} ci_low {r['ci_low']:.4f} does not clear {args.target}")

    gate("submission", submission)
    if verification:
        gate("verification", verification)
    else:
        print("  (no verification/ rerun present yet — append one before merging)")

    print(f"\n{'VERDICT: PASS' if not fails.items else f'VERDICT: FAIL ({len(fails.items)} problem(s))'}")
    return 0 if not fails.items else 1


if __name__ == "__main__":
    raise SystemExit(main())
