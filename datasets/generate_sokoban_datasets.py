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
# Train layout (TRAIN_SIZE rows):
#   rows [0, IGNITION):           easy only — high mixed fraction from step 0, short
#                                 rollouts -> fast steps while LR is at peak
#   rows [IGNITION, RAMP_END):    seeded interleave, P(stable) ramps RAMP_P0 -> RAMP_P1
#   rows [RAMP_END, TRAIN_SIZE):  core: stable / stretch (2-3 box, >=9 moves) mix for
#                                 future longer/faster record attempts
#
# Eval layout (EVAL_SIZE rows): 2-box stable-band buckets by reference move count, shares
# calibrated so base pass@1 ~0.15 and the trained ceiling (ckpt-C pass@16) is ~0.9.
#
# All seeds are fixed; the output is deterministic. reasoning_gym streams puzzles per-item
# as ~(seed+index), so band seeds are spaced 1M+ apart and the eval is additionally made
# disjoint from train by an explicit gamestr exclude filter (shared across all pools).

TRAIN_SIZE = 10_000
EVAL_SIZE = 256

IGNITION = 500       # rows of pure easy at the head of the file
RAMP_END = 2_000     # interleave ends here; ~the horizon of a 1-hour run
RAMP_P0, RAMP_P1 = 0.10, 0.80   # P(stable) at the start/end of the ramp
CORE_P_STRETCH = 0.20           # stretch share in rows [RAMP_END, TRAIN_SIZE)

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
EASY = Band("easy", seed=42, min_boxes=1, max_boxes=1, max_depth=8, min_moves=3, max_moves=8)
STABLE = Band("stable", seed=1_000_000, min_boxes=2, max_boxes=2, max_depth=10, min_moves=4,
              max_moves=10)
STRETCH = Band("stretch", seed=2_000_000, min_boxes=2, max_boxes=3, max_depth=12,
               max_w=8, max_h=8, min_moves=9)

# Eval bands (far seeds; additionally excluded from train by construction).
EVAL_CORE = Band("eval_core", seed=10_000_000, min_boxes=2, max_boxes=2, max_depth=10,
                 min_moves=5)
EVAL_HEADROOM = Band("eval_headroom", seed=11_000_000, min_boxes=2, max_boxes=2,
                     max_depth=12, min_moves=11)

# Eval bucket shares (by reference move count), n=EVAL_SIZE total.
# Calibration (strict protocol, k=16): base pass@1 / ckpt-C pass@16 per 2-box bucket:
#   5-6: 0.29 / 0.98   7-8: 0.11 / 0.89   9-10: 0.06 / 0.87   11+: 0.05 / 0.60
# Measured on the built eval: base pass@1 0.140, trained pass@16 0.906 strict / 0.969 generous.
EVAL_SHARES = {"5-6": 77, "7-8": 102, "9-10": 64, "11+": 13}


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

    drain(EVAL_CORE, {"5-6", "7-8", "9-10"}, max_candidates=60_000)
    if EVAL_SHARES.get("11+", 0) > 0:
        drain(EVAL_HEADROOM, {"11+"}, max_candidates=60_000)
    rng = random.Random(ASSEMBLY_SEED)
    rng.shuffle(picked)  # so --eval-limit subsets stay representative
    return picked


def assemble_official_train(easy: list, stable: list, stretch: list) -> list[dict[str, Any]]:
    rng = random.Random(ASSEMBLY_SEED + 1)
    easy_it, stable_it, stretch_it = iter(easy), iter(stable), iter(stretch)
    rows: list[dict[str, Any]] = []
    for _ in range(IGNITION):
        rows.append(next(easy_it))
    ramp_n = RAMP_END - IGNITION
    for i in range(ramp_n):
        p_stable = RAMP_P0 + (RAMP_P1 - RAMP_P0) * (i / max(1, ramp_n - 1))
        rows.append(next(stable_it) if rng.random() < p_stable else next(easy_it))
    for _ in range(TRAIN_SIZE - RAMP_END):
        rows.append(next(stretch_it) if rng.random() < CORE_P_STRETCH else next(stable_it))
    return rows


def official_pool_needs(seed: int = ASSEMBLY_SEED + 1) -> dict[str, int]:
    """Dry-run the assembly rng to compute exact per-band counts."""
    rng = random.Random(seed)
    counts = {"easy": IGNITION, "stable": 0, "stretch": 0}
    ramp_n = RAMP_END - IGNITION
    for i in range(ramp_n):
        p_stable = RAMP_P0 + (RAMP_P1 - RAMP_P0) * (i / max(1, ramp_n - 1))
        counts["stable" if rng.random() < p_stable else "easy"] += 1
    for _ in range(TRAIN_SIZE - RAMP_END):
        counts["stretch" if rng.random() < CORE_P_STRETCH else "stable"] += 1
    return counts


def summarize_official(name: str, rows: list[dict[str, Any]]) -> None:
    from collections import Counter
    buckets = Counter(bucket_of(reference_move_count(e)) for e in rows)
    bands = Counter(e["metadata"]["band"] for e in rows)
    head = Counter(e["metadata"]["band"] for e in rows[:RAMP_END]) if name == "train" else None
    print(f"{name}: n={len(rows)} buckets={dict(sorted(buckets.items()))} "
          f"bands={dict(sorted(bands.items()))}"
          + (f" first{RAMP_END}={dict(sorted(head.items()))}" if head else ""), flush=True)


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
    easy = generate_band_pool(EASY, needs["easy"], exclude=seen)
    stable = generate_band_pool(STABLE, needs["stable"], exclude=seen)
    stretch = generate_band_pool(STRETCH, needs["stretch"], exclude=seen)
    train_rows = assemble_official_train(easy, stable, stretch)
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
