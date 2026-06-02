"""Generate fixed Reasoning Gym Sokoban train/eval JSONL datasets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
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
) -> list[dict[str, Any]]:
    if size < 1:
        raise ValueError("split size must be at least 1")
    candidate_size = size
    if min_moves > 0:
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
    for i in range(len(dataset)):
        entry = clean_entry(dict(dataset[i]))
        if "gamestr" not in entry["metadata"]:
            raise ValueError(f"generated entry {i} is missing metadata.gamestr")
        if reference_move_count(entry) < min_moves:
            continue
        moves = normalize_sokoban_moves(entry["answer"])
        if moves is None or scorer.score_answer(answer=moves, entry=entry) != 1.0:
            continue
        entries.append(entry)
        if len(entries) >= size:
            break

    if len(entries) < size:
        raise ValueError(
            f"only generated {len(entries)} examples after min-moves={min_moves}; "
            "increase --max-candidates, lower the move floor, or relax puzzle difficulty"
        )
    return entries


def write_jsonl(path: Path, entries: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_verification(train_output: Path, eval_output: Path) -> None:
    runner = Path(__file__).resolve().parent / "run_rl.py"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fixed Sokoban JSONL datasets for run_rl.py")
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=43)
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
    parser.add_argument("--train-output", type=Path, default=Path("datasets/sokoban_train.jsonl"))
    parser.add_argument("--eval-output", type=Path, default=Path("datasets/sokoban_eval.jsonl"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run run_rl.py --verify-datasets-only after writing the files",
    )
    return parser


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
        existing = [str(path) for path in (args.train_output, args.eval_output) if path.exists()]
        if existing:
            raise FileExistsError(
                "output file already exists; pass --overwrite to replace: " + ", ".join(existing)
            )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

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
    )

    write_jsonl(args.train_output, train_entries, overwrite=args.overwrite)
    write_jsonl(args.eval_output, eval_entries, overwrite=args.overwrite)
    print(f"Wrote train={len(train_entries)} to {args.train_output}", flush=True)
    print(f"Wrote eval={len(eval_entries)} to {args.eval_output}", flush=True)

    if args.verify:
        run_verification(args.train_output, args.eval_output)


if __name__ == "__main__":
    main()
