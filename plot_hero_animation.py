"""Generate the top-level README hero animation from a record directory.

Usage:
    uv run --with matplotlib --with pandas --with pillow \
        python plot_hero_animation.py records/2026-06-17_01_grpo

The GIF is intentionally not a leaderboard. It is a quick visual hook: training
solve rate climbs on the left; the held-out pass@1 scorecard on the right is the
actual benchmark verdict.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from PIL import Image  # noqa: E402

from make_record_report import (  # noqa: E402
    BG,
    GRID,
    INK,
    MUTED,
    SEED_COLORS,
    TARGET_C,
    glow,
    parse_train_log,
    set_house_style,
    smooth,
)


def hms(seconds: float) -> str:
    s = int(seconds)
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}"


def load_record(record_dir: Path) -> tuple[dict, dict]:
    logs = sorted(p for p in record_dir.glob("train_log_seed*.txt") if not p.name.endswith(".flops.txt"))
    evals = sorted(record_dir.glob("eval_seed*.json"))
    if not logs or not evals:
        raise SystemExit(f"{record_dir}: need train_log_seed*.txt and eval_seed*.json")
    return parse_train_log(logs[0]), json.loads(evals[0].read_text())


def draw_frame(
    frame: int,
    total_frames: int,
    log: dict,
    eval_result: dict,
    target: float,
    out_png: Path | None = None,
):
    df = log["df"]
    x = df.record_time_s / 60.0
    raw_y = df.solved_frac
    y = smooth(raw_y)
    n = max(2, round((frame + 1) / total_frames * len(df)))
    final_time = log["final_time_s"]
    final_pass = eval_result["pass_at_1"]

    fig = plt.figure(figsize=(10.8, 6.1), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.52, 1.0], left=0.075, right=0.965,
                          top=0.80, bottom=0.13, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    card = fig.add_subplot(gs[0, 1])
    card.axis("off")

    fig.text(0.075, 0.925, "Sokoban Speedrun", fontsize=33, fontweight="bold",
             color=INK, ha="left", va="center")

    ax.plot(x.iloc[:n], raw_y.iloc[:n], color=SEED_COLORS[0], alpha=0.13, lw=1.2)
    line = ax.plot(x.iloc[:n], y.iloc[:n], color=SEED_COLORS[0], lw=3.0)[0]
    glow(line, lw=9, alpha=0.28)
    ax.fill_between(x.iloc[:n], y.iloc[:n], color=SEED_COLORS[0], alpha=0.13)
    ax.scatter([x.iloc[n - 1]], [y.iloc[n - 1]], s=58, color=SEED_COLORS[0],
               edgecolor=BG, linewidth=1.5, zorder=5)

    ax.set_xlim(0, max(90, float(x.max()) * 1.03))
    ax.set_ylim(0.32, max(0.78, float(y.max()) * 1.08))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("minutes")
    ax.set_ylabel("train solve rate")
    ax.grid(True, axis="y", color=GRID, alpha=0.55)

    card.text(0.0, 0.82, "RECORD", color=MUTED, fontsize=14,
              fontweight="bold", va="top")
    card.text(0.0, 0.72, hms(final_time), color=INK, fontsize=42,
              fontweight="bold", va="top")

    reveal_frame = int(total_frames * 0.72)
    revealed = frame >= reveal_frame
    if revealed:
        card.text(0.0, 0.45, "PASS@1", color=MUTED, fontsize=14,
                  fontweight="bold", va="top")
        card.text(0.0, 0.36, f"{final_pass:.1%}", color=SEED_COLORS[0], fontsize=64,
                  fontweight="bold", va="top")
        card.text(0.0, 0.12, f"target > {target:.0%}", color=TARGET_C, fontsize=18,
                  fontweight="bold", va="top")

    if out_png is not None:
        fig.savefig(out_png, dpi=120)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_dir", nargs="?", type=Path, default=Path("records/2026-06-17_01_grpo"))
    parser.add_argument("--out", type=Path, default=Path("records/hero.gif"))
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument("--frames", type=int, default=76)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    set_house_style()
    log, eval_result = load_record(args.record_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    for frame in range(args.frames):
        fig = draw_frame(frame, args.frames, log, eval_result, args.target)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    duration_ms = round(1000 / args.fps)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )

    plt.close("all")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
