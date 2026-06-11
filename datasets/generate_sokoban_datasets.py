"""Generate fixed Reasoning Gym Sokoban train/eval JSONL datasets."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import reasoning_gym


SOKOBAN_MOVE_STRING_RE = re.compile(r"[UDLR]+")


def normalize_sokoban_moves(candidate: str) -> str | None:
    compact = "".join(candidate.split()).upper()
    if compact and SOKOBAN_MOVE_STRING_RE.fullmatch(compact):
        return compact
    return None


def reference_move_count(entry: dict[str, Any]) -> int:
    moves = normalize_sokoban_moves(entry.get("answer") or "")
    return len(moves) if moves else 0


def build_scorer():
    return reasoning_gym.create_dataset("sokoban", size=1, seed=0)


def normalize_board_text(board: str) -> str:
    """Keep only the Sokoban grid, with one space between cells and no blank lines."""
    rows = [" ".join(line.split()) for line in board.splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty Sokoban board")
    return "\n".join(rows)


def extract_board(entry: dict[str, Any]) -> str:
    metadata = dict(entry.get("metadata") or {})
    gamestr = metadata.get("gamestr")
    if isinstance(gamestr, str) and gamestr.strip():
        return normalize_board_text(gamestr)
    question = entry.get("question")
    if not isinstance(question, str):
        raise ValueError("generated entry is missing question text")
    return normalize_board_text(question.split("Here is your puzzle:", 1)[-1])


def clean_entry(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(entry.get("metadata") or {})
    return {
        "question": extract_board(entry),
        "answer": entry["answer"],
        "metadata": metadata,
    }


def generate_split(
    *,
    size: int,
    seed: int,
    min_w: int,
    max_w: int,
    min_h: int,
    max_h: int,
    min_boxes: int,
    max_boxes: int,
    max_depth: int,
    min_moves: int,
    oversample_factor: int,
    max_candidates: int,
    exclude: set[str] | None = None,
) -> list[dict[str, Any]]:
    if size < 1:
        raise ValueError("split size must be at least 1")
    candidate_size = size
    # Oversample to leave headroom for the min-moves, solvability, and exclude (disjointness) filters.
    if min_moves > 0 or exclude:
        candidate_size = min(max(size * oversample_factor, size), max_candidates)
    if candidate_size < size:
        raise ValueError("--max-candidates must be at least the requested split size")

    dataset = reasoning_gym.create_dataset(
        "sokoban",
        size=candidate_size,
        seed=seed,
        min_w=min_w,
        max_w=max_w,
        min_h=min_h,
        max_h=max_h,
        min_boxes=min_boxes,
        max_boxes=max_boxes,
        max_depth=max_depth,
    )
    scorer = build_scorer()
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i in range(len(dataset)):
        entry = clean_entry(dict(dataset[i]))
        if "gamestr" not in entry["metadata"]:
            raise ValueError(f"generated entry {i} is missing metadata.gamestr")
        board_key = normalize_board_text(entry["metadata"]["gamestr"])
        if board_key in seen:
            continue
        if reference_move_count(entry) < min_moves:
            continue
        moves = normalize_sokoban_moves(entry["answer"])
        if moves is None or scorer.score_answer(answer=moves, entry=entry) != 1.0:
            continue
        if exclude is not None and board_key in exclude:
            continue  # incidental collision with the excluded (e.g. train) split — keep splits disjoint
        entries.append(entry)
        seen.add(board_key)
        if len(entries) >= size:
            break

    if len(entries) < size:
        raise ValueError(
            f"only generated {len(entries)} examples after min-moves={min_moves}/exclude; "
            "increase --max-candidates, lower the move floor, relax puzzle difficulty, "
            "or allow a larger board/box search space"
        )
    return entries


def write_jsonl(path: Path, entries: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _gamestr_set(path: Path, entries: list[dict[str, Any]] | None) -> set[str]:
    """Normalized gamestr keys for a split, from in-memory entries or by reading the file."""
    if entries is not None:
        return {normalize_board_text(e["metadata"]["gamestr"]) for e in entries}
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                keys.add(normalize_board_text(json.loads(line)["metadata"]["gamestr"]))
    return keys


def assert_disjoint(
    args: argparse.Namespace,
    train_entries: list[dict[str, Any]] | None,
    eval_entries: list[dict[str, Any]] | None,
) -> None:
    """Assert no puzzle (by normalized gamestr) appears in both train and eval."""
    if not args.train_output.exists() or not args.eval_output.exists():
        print("skip --assert-disjoint: both split files must exist", flush=True)
        return
    train_keys = _gamestr_set(args.train_output, train_entries)
    eval_keys = _gamestr_set(args.eval_output, eval_entries)
    overlap = train_keys & eval_keys
    if overlap:
        raise ValueError(
            f"train/eval overlap on {len(overlap)} puzzle(s); splits must be disjoint "
            f"(use distinct --train-seed/--eval-seed)"
        )
    print(f"disjoint OK: train={len(train_keys)} eval={len(eval_keys)} share 0 puzzles", flush=True)


def run_verification(train_output: Path, eval_output: Path) -> None:
    runner = Path(__file__).resolve().parent.parent / "speedrun.py"
    command = [
        sys.executable,
        str(runner),
        "--verify-datasets-only",
        "--train-data",
        str(train_output),
        "--eval-data",
        str(eval_output),
    ]
    subprocess.run(command, check=True)


# ====================================== OFFICIAL MIX ======================================
# The official leaderboard datasets (--official). Everything here is a FIXED constant: the
# published files must be reproducible exactly, so there are no CLI knobs for the mix.
#
#
# Train layout (TRAIN_SIZE rows) — mixture probabilities are piecewise-linear and
# CONTINUOUS at every phase boundary. The previous layout had a double cliff at row
# 2,000 (P(stable) 0.80 -> 1.0 and stretch 0 -> 20% in one row); crossing it measurably
# sagged online solve (~0.62 -> 0.53) and tripled all-fail group waste (rollout-fitted
# per-band rates: at first contact the 11-14-move stretch half solves at ~0.2-0.3).
#
#   rows [0, IGNITION):                 ignite only (2-box 4-6 movers, base solve ~0.29):
#                                       variance-rich AND on the eval distribution from step 0
#   rows [IGNITION, HANDOVER_END):      P(stable) ramps RAMP_P0 -> 1.0; ignite is FULLY retired
#                                       at the boundary
#   rows [HANDOVER_END, STRETCH_RAMP_END): P(stretch) ramps 0 -> CORE_P_STRETCH and
#                                       P(stable_long) ramps 0 -> CORE_P_LONG (frontier and the
#                                       eval's 11+ bucket blend in as skill grows, never as a step)
#   rows [STRETCH_RAMP_END, LATE_RAMP_START): hold at CORE_P_STRETCH / CORE_P_LONG
#   rows [LATE_RAMP_START, TRAIN_SIZE): P(stretch) ramps to LATE_P_STRETCH — keeps per-group
#                                       reward variance alive for long runs that saturate stable
#
# Eval layout (EVAL_SIZE rows): 2-box stable-band buckets by reference move count, shares
# calibrated so base pass@1 ~0.15 and the trained ceiling (ckpt-C pass@16) is ~0.9.
# (The eval build is independent of the train schedule: changing the constants below
# cannot move a single eval row.)
#
# All seeds are fixed; the output is deterministic. reasoning_gym streams puzzles per-item
# as ~(seed+index), so band seeds are spaced 1M+ apart and the eval is additionally made
# disjoint from train by an explicit gamestr exclude filter (shared across all pools).

TRAIN_SIZE = 10_000
EVAL_SIZE = 384

# Pacing is calibrated to the ~100-step record horizon (a 100-step run consumes ~2,300 rows
# at ~23 accepted-groups-and-rejects per step): pure 2-box training must arrive with enough
# steps left to matter. v6 (handover at 2,200 = step ~95) spent steps 45-65 on ~50% easy rows
# it already solved at 0.66-0.73 and reached pure stable 5 steps before eval — pass@1 0.570 vs
# v5's 0.616. This pacing gives ~35 pure-stable steps + ~16% stretch by step 100 (the eval's
# 9+-move buckets are 30% of puzzles) while keeping every transition continuous.
IGNITION = 300            # rows of pure easy at the head of the file (~13 steps; LR warmup is 5)
HANDOVER_END = 1_500      # P(stable) reaches 1.0 here (~step 65 of a record run)
RAMP_P0 = 0.10            # P(stable) at the start of the handover ramp
STRETCH_RAMP_END = 2_500  # P(stretch) reaches CORE_P_STRETCH here
CORE_P_STRETCH = 0.20     # stretch share in the core plateau
LATE_RAMP_START = 6_000   # late frontier ramp starts here ...
LATE_P_STRETCH = 0.35     # ... ending at this stretch share by the end of the file

ASSEMBLY_SEED = 0x50C0BA  # rng for the interleave + eval shuffle


@dataclass(frozen=True)
class Band:
    name: str
    seed: int
    min_boxes: int
    max_boxes: int
    max_depth: int
    min_w: int = 6
    max_w: int = 7
    min_h: int = 6
    max_h: int = 7
    min_moves: int = 0
    max_moves: int = 10**9


# Train bands. Seeds are >=1M apart (reasoning_gym streams ~(seed+index) per item).
# max_depth bounds the generator's search, NOT the solution length (depth-8 emits some
# 9-10 move solutions), so the probed band envelopes need explicit max_moves caps.
# IGNITE replaced the old 1-box easy band: 2-box 4-6-movers solve at ~0.29 base — already in
# the variance sweet band at cold start — so every ignition gradient is on the eval
# distribution instead of a 1-box scaffold that gets retired anyway.
# STABLE_LONG covers the eval's 9-10/11+ buckets in-distribution (2-box, small boards);
# stretch alone confounded longer solutions with 3-box/8x8 transfer.
IGNITE = Band("ignite", seed=42, min_boxes=2, max_boxes=2, max_depth=8, min_moves=4, max_moves=6)
STABLE = Band("stable", seed=1_000_000, min_boxes=2, max_boxes=2, max_depth=10, min_moves=4,
              max_moves=10)
STRETCH = Band("stretch", seed=2_000_000, min_boxes=2, max_boxes=3, max_depth=12,
               max_w=8, max_h=8, min_moves=9)
STABLE_LONG = Band("stable_long", seed=3_000_000, min_boxes=2, max_boxes=2, max_depth=16,
                   min_moves=11, max_moves=16)

# Eval bands (far seeds; additionally excluded from train by construction).
EVAL_CORE = Band("eval_core", seed=10_000_000, min_boxes=2, max_boxes=2, max_depth=10,
                 min_moves=5)
EVAL_HEADROOM = Band("eval_headroom", seed=11_000_000, min_boxes=2, max_boxes=2,
                     max_depth=12, min_moves=11)

# Eval bucket shares (by reference move count), n=EVAL_SIZE total.
# Calibration: base pass@1 per 2-box bucket (probed): 5-6: 0.29  7-8: 0.11  9-10: 0.06  11+: 0.05.
# v6-recipe (100 steps, 2026-06-10) per-bucket pass@1 / pass@16, measured from eval rollouts:
#   5-6: 0.734/0.961   7-8: 0.567/0.951   9-10: 0.415/0.922   11+: 0.375/0.923
# These shares weight the hard buckets (where future records differentiate) while projecting
# base ~0.141 and current-recipe pass@1 ~0.553 — comfortably above the 0.50 TARGET, ceiling ~0.94.
EVAL_SHARES = {"5-6": 108, "7-8": 132, "9-10": 104, "11+": 40}


def bucket_of(moves: int) -> str:
    if moves <= 4:
        return "<=4"
    if moves <= 6:
        return "5-6"
    if moves <= 8:
        return "7-8"
    if moves <= 10:
        return "9-10"
    return "11+"


def generate_band_pool(band: Band, need: int, *, exclude: set[str],
                       max_candidates: int = 120_000) -> list[dict[str, Any]]:
    """Generate `need` verified, deduped, non-excluded puzzles from a band.

    Mirrors generate_split but takes the band spec, a max-moves cap, and a shared
    cross-band exclude set (mutated as puzzles are kept, so every pool built with the
    same set is mutually disjoint).
    """
    scorer = build_scorer()
    entries: list[dict[str, Any]] = []
    # Stream in chunks so easy bands don't pay for the worst-case candidate count.
    chunk, offset = 4 * need, 0
    while len(entries) < need:
        if offset >= max_candidates:
            raise ValueError(
                f"band {band.name}: only {len(entries)}/{need} after {offset} candidates; "
                "raise max_candidates or relax the band"
            )
        size = min(chunk, max_candidates - offset)
        dataset = reasoning_gym.create_dataset(
            "sokoban", size=size, seed=band.seed + offset,
            min_w=band.min_w, max_w=band.max_w, min_h=band.min_h, max_h=band.max_h,
            min_boxes=band.min_boxes, max_boxes=band.max_boxes, max_depth=band.max_depth,
        )
        for i in range(len(dataset)):
            entry = clean_entry(dict(dataset[i]))
            board_key = normalize_board_text(entry["metadata"]["gamestr"])
            if board_key in exclude:
                continue
            moves_n = reference_move_count(entry)
            if not (band.min_moves <= moves_n <= band.max_moves):
                continue
            moves = normalize_sokoban_moves(entry["answer"])
            if moves is None or scorer.score_answer(answer=moves, entry=entry) != 1.0:
                continue
            entry["metadata"]["band"] = band.name
            entry["metadata"]["ref_moves"] = moves_n
            entries.append(entry)
            exclude.add(board_key)
            if len(entries) >= need:
                break
        offset += size
    print(f"band {band.name}: {len(entries)} puzzles from {offset} candidates "
          f"(seed {band.seed})", flush=True)
    return entries


def build_official_eval(exclude: set[str]) -> list[dict[str, Any]]:
    """Bucketed eval: draw each move-count bucket to its calibrated share."""
    need_by_bucket = dict(EVAL_SHARES)
    picked: list[dict[str, Any]] = []

    def drain(band: Band, want_buckets: set[str], max_candidates: int) -> None:
        scorer = build_scorer()
        chunk, offset = 2_000, 0
        while any(need_by_bucket[b] > 0 for b in want_buckets):
            if offset >= max_candidates:
                raise ValueError(f"eval band {band.name}: still need {need_by_bucket} "
                                 f"after {offset} candidates")
            dataset = reasoning_gym.create_dataset(
                "sokoban", size=chunk, seed=band.seed + offset,
                min_w=band.min_w, max_w=band.max_w, min_h=band.min_h, max_h=band.max_h,
                min_boxes=band.min_boxes, max_boxes=band.max_boxes, max_depth=band.max_depth,
            )
            for i in range(len(dataset)):
                entry = clean_entry(dict(dataset[i]))
                board_key = normalize_board_text(entry["metadata"]["gamestr"])
                if board_key in exclude:
                    continue
                moves_n = reference_move_count(entry)
                bucket = bucket_of(moves_n)
                if bucket not in want_buckets or need_by_bucket.get(bucket, 0) <= 0:
                    continue
                if moves_n < band.min_moves:
                    continue
                moves = normalize_sokoban_moves(entry["answer"])
                if moves is None or scorer.score_answer(answer=moves, entry=entry) != 1.0:
                    continue
                entry["metadata"]["band"] = band.name
                entry["metadata"]["ref_moves"] = moves_n
                picked.append(entry)
                exclude.add(board_key)
                need_by_bucket[bucket] -= 1
            offset += chunk

    drain(EVAL_CORE, {"5-6", "7-8", "9-10"}, max_candidates=120_000)
    if EVAL_SHARES.get("11+", 0) > 0:
        drain(EVAL_HEADROOM, {"11+"}, max_candidates=200_000)  # 11+ movers are rare in-stream
    rng = random.Random(ASSEMBLY_SEED)
    rng.shuffle(picked)  # so --eval-limit subsets stay representative
    return picked


CORE_P_LONG = 0.08        # stable_long share: ramps 0 -> this over [HANDOVER_END, STRETCH_RAMP_END),
                          # then holds — direct in-distribution coverage of the eval's 11+ bucket

TRAIN_BAND_NAMES = ("ignite", "stable", "stretch", "stable_long")


def band_schedule(rng: random.Random) -> list[str]:
    """One band name per train row — the SINGLE definition of the curriculum ordering.

    Both pool sizing and assembly consume this (same seed => same draws), so the two can
    never drift apart. Mixture probabilities are piecewise-linear and continuous at every
    boundary; see the layout comment above for why cliffs are forbidden."""
    names: list[str] = []
    for i in range(TRAIN_SIZE):
        if i < IGNITION:
            names.append("ignite")
        elif i < HANDOVER_END:
            f = (i - IGNITION) / max(1, HANDOVER_END - 1 - IGNITION)
            p_stable = RAMP_P0 + (1.0 - RAMP_P0) * f
            names.append("stable" if rng.random() < p_stable else "ignite")
        else:
            if i < STRETCH_RAMP_END:
                g = (i - HANDOVER_END) / max(1, STRETCH_RAMP_END - HANDOVER_END)
                p_str, p_long = CORE_P_STRETCH * g, CORE_P_LONG * g
            elif i < LATE_RAMP_START:
                p_str, p_long = CORE_P_STRETCH, CORE_P_LONG
            else:
                p_str = CORE_P_STRETCH + (LATE_P_STRETCH - CORE_P_STRETCH) * (
                    (i - LATE_RAMP_START) / max(1, TRAIN_SIZE - 1 - LATE_RAMP_START))
                p_long = CORE_P_LONG
            u = rng.random()
            names.append("stretch" if u < p_str
                         else "stable_long" if u < p_str + p_long
                         else "stable")
    return names


def assemble_official_train(pools_by_band: dict[str, list]) -> list[dict[str, Any]]:
    pools = {name: iter(pool) for name, pool in pools_by_band.items()}
    return [next(pools[name]) for name in band_schedule(random.Random(ASSEMBLY_SEED + 1))]


def official_pool_needs() -> dict[str, int]:
    """Dry-run the schedule rng to compute exact per-band counts."""
    from collections import Counter
    counts = Counter(band_schedule(random.Random(ASSEMBLY_SEED + 1)))
    return {name: counts[name] for name in TRAIN_BAND_NAMES}


def summarize_official(name: str, rows: list[dict[str, Any]]) -> None:
    from collections import Counter
    buckets = Counter(bucket_of(reference_move_count(e)) for e in rows)
    bands = Counter(e["metadata"]["band"] for e in rows)
    head = Counter(e["metadata"]["band"] for e in rows[:HANDOVER_END]) if name == "train" else None
    print(f"{name}: n={len(rows)} buckets={dict(sorted(buckets.items()))} "
          f"bands={dict(sorted(bands.items()))}"
          + (f" first{HANDOVER_END}={dict(sorted(head.items()))}" if head else ""), flush=True)


def build_official_mix(args: argparse.Namespace) -> None:
    """--official entrypoint: build both official files from the fixed constants above."""
    if sum(EVAL_SHARES.values()) != EVAL_SIZE:
        raise ValueError(f"EVAL_SHARES sums to {sum(EVAL_SHARES.values())}, want {EVAL_SIZE}")
    if not args.overwrite:
        existing = [str(p) for p in (args.train_output, args.eval_output) if p.exists()]
        if existing:  # fail BEFORE the ~10-minute generation, not at write time
            raise FileExistsError(
                "output file already exists; pass --overwrite to replace: " + ", ".join(existing)
            )

    seen: set[str] = set()  # shared across ALL pools => train/eval disjoint by construction
    # Eval first so its (far-seed) puzzles can never be displaced by train collisions.
    eval_rows = build_official_eval(seen)
    summarize_official("eval", eval_rows)

    needs = official_pool_needs()
    print(f"train pool needs: {needs}", flush=True)
    pools = {
        "ignite": generate_band_pool(IGNITE, needs["ignite"], exclude=seen),
        "stable": generate_band_pool(STABLE, needs["stable"], exclude=seen),
        "stretch": generate_band_pool(STRETCH, needs["stretch"], exclude=seen),
        # 11-16-move 2-box puzzles are rare in the generator stream; give the drain headroom.
        "stable_long": generate_band_pool(STABLE_LONG, needs["stable_long"], exclude=seen,
                                          max_candidates=600_000),
    }
    train_rows = assemble_official_train(pools)
    if len(train_rows) != TRAIN_SIZE:
        raise AssertionError(f"assembled {len(train_rows)} train rows, want {TRAIN_SIZE}")
    summarize_official("train", train_rows)

    train_keys = {normalize_board_text(e["metadata"]["gamestr"]) for e in train_rows}
    eval_keys = {normalize_board_text(e["metadata"]["gamestr"]) for e in eval_rows}
    overlap = train_keys & eval_keys
    if overlap:
        raise AssertionError(f"train/eval overlap on {len(overlap)} puzzles")
    print(f"disjoint OK: train={len(train_keys)} eval={len(eval_keys)} share 0", flush=True)

    write_jsonl(args.eval_output, eval_rows, overwrite=args.overwrite)
    write_jsonl(args.train_output, train_rows, overwrite=args.overwrite)
    print(f"Wrote {args.train_output} and {args.eval_output}", flush=True)

    if args.verify:
        run_verification(args.train_output, args.eval_output)

# ==========================================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fixed Sokoban JSONL datasets for speedrun.py")
    parser.add_argument(
        "--official",
        action="store_true",
        help="Build the OFFICIAL leaderboard mix (curriculum-ordered 10k train + calibrated "
        "256 eval) from the fixed constants in this file. Ignores the band/size/seed flags "
        "below; honors --train-output/--eval-output/--overwrite/--verify.",
    )
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=42)
    # reasoning_gym seeds per item as ~(seed+index), so adjacent seeds yield near-identical puzzle
    # streams: eval-seed 43 overlapped train-seed 42 (10k items => effective seeds ~42..10041) on
    # ~99% of puzzles. The eval seed MUST be far outside [train_seed, train_seed+train_size]; use a
    # large value (verify with --assert-disjoint).
    parser.add_argument("--eval-seed", type=int, default=10_000_000)
    parser.add_argument("--min-w", type=int, default=6)
    parser.add_argument("--max-w", type=int, default=8)
    parser.add_argument("--min-h", type=int, default=6)
    parser.add_argument("--max-h", type=int, default=8)
    parser.add_argument("--min-boxes", type=int, default=2)
    parser.add_argument("--max-boxes", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--train-min-moves", type=int, default=0)
    parser.add_argument("--eval-min-moves", type=int, default=0)
    parser.add_argument("--oversample-factor", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=20_000)
    parser.add_argument("--train-output", type=Path, default=None)
    parser.add_argument("--eval-output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--splits",
        choices=["both", "train", "eval"],
        default="both",
        help="Which split(s) to (re)generate. 'eval' regenerates only the held-out set, "
        "leaving the published train file untouched.",
    )
    parser.add_argument(
        "--assert-disjoint",
        action="store_true",
        help="After writing, assert no metadata.gamestr is shared between the train and eval files.",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run speedrun.py --verify-datasets-only after writing the files",
    )
    return parser


def resolve_output_defaults(args: argparse.Namespace) -> None:
    if args.train_output is None:
        args.train_output = Path("datasets/sokoban_train.jsonl")
    if args.eval_output is None:
        args.eval_output = Path("datasets/sokoban_eval.jsonl")


def validate_args(args: argparse.Namespace) -> None:
    if args.train_size < 1:
        raise ValueError("--train-size must be at least 1")
    if args.eval_size < 1:
        raise ValueError("--eval-size must be at least 1")
    if args.min_w > args.max_w:
        raise ValueError("--min-w must be <= --max-w")
    if args.min_h > args.max_h:
        raise ValueError("--min-h must be <= --max-h")
    if args.min_boxes > args.max_boxes:
        raise ValueError("--min-boxes must be <= --max-boxes")
    if args.max_depth <= 1:
        raise ValueError("--max-depth must be greater than 1")
    if args.train_min_moves < 0 or args.eval_min_moves < 0:
        raise ValueError("min-moves filters must be non-negative")
    if args.oversample_factor < 1:
        raise ValueError("--oversample-factor must be at least 1")
    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be at least 1")
    if args.train_output == args.eval_output:
        raise ValueError("--train-output and --eval-output must be different files")
    if not args.overwrite:
        targets = []
        if args.splits in ("both", "train"):
            targets.append(args.train_output)
        if args.splits in ("both", "eval"):
            targets.append(args.eval_output)
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "output file already exists; pass --overwrite to replace: " + ", ".join(existing)
            )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    resolve_output_defaults(args)

    if args.official:
        if args.train_output == args.eval_output:
            raise ValueError("--train-output and --eval-output must be different files")
        build_official_mix(args)
        return

    validate_args(args)

    train_entries: list[dict[str, Any]] | None = None
    eval_entries: list[dict[str, Any]] | None = None

    if args.splits in ("both", "train"):
        train_entries = generate_split(
            size=args.train_size,
            seed=args.train_seed,
            min_w=args.min_w,
            max_w=args.max_w,
            min_h=args.min_h,
            max_h=args.max_h,
            min_boxes=args.min_boxes,
            max_boxes=args.max_boxes,
            max_depth=args.max_depth,
            min_moves=args.train_min_moves,
            oversample_factor=args.oversample_factor,
            max_candidates=args.max_candidates,
        )
        write_jsonl(args.train_output, train_entries, overwrite=args.overwrite)
        print(f"Wrote train={len(train_entries)} to {args.train_output}", flush=True)

    if args.splits in ("both", "eval"):
        # Make eval disjoint from train BY CONSTRUCTION: exclude any puzzle (by normalized gamestr)
        # that appears in train. A far --eval-seed avoids the bulk seed-range overlap, but a few
        # incidental collisions remain at 10k+2k scale; this filter drops them as they're generated.
        exclude = None
        if train_entries is not None:
            exclude = _gamestr_set(args.train_output, train_entries)
        elif args.train_output.exists():
            exclude = _gamestr_set(args.train_output, None)
        eval_entries = generate_split(
            size=args.eval_size,
            seed=args.eval_seed,
            min_w=args.min_w,
            max_w=args.max_w,
            min_h=args.min_h,
            max_h=args.max_h,
            min_boxes=args.min_boxes,
            max_boxes=args.max_boxes,
            max_depth=args.max_depth,
            min_moves=args.eval_min_moves,
            oversample_factor=args.oversample_factor,
            max_candidates=args.max_candidates,
            exclude=exclude,
        )
        write_jsonl(args.eval_output, eval_entries, overwrite=args.overwrite)
        print(f"Wrote eval={len(eval_entries)} to {args.eval_output}", flush=True)

    if args.assert_disjoint:
        assert_disjoint(args, train_entries, eval_entries)

    if args.verify:
        run_verification(args.train_output, args.eval_output)


if __name__ == "__main__":
    main()
