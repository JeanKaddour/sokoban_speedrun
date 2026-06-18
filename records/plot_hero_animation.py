"""Generate the top-level README hero animation from a record directory.

Usage:
    # no arg: auto-targets the current record (fastest run under records/ that clears
    # the target), so regenerating after a new submission needs no edits:
    uv run --with matplotlib --with pandas --with pillow \
        python records/plot_hero_animation.py

    # or pin a specific record dir:
    uv run ... python records/plot_hero_animation.py records/2026-06-17_01_grpo \
        --wandb-runs entity/project/run_id

    # dump a single still (for eyeballing the composition) instead of the GIF:
    uv run ... python records/plot_hero_animation.py records/<dir> --still records/hero_still.png

The GIF is intentionally not a leaderboard. It is a quick visual hook: the train solve
rate climbs on the left while a stopwatch ticks up the record clock on the right
— same clock as the x-axis — and the held-out pass@1 scorecard counts up
from the base model to the final record, the actual benchmark verdict.

The README already carries the "Sokoban Speedrun" H1, so the animation deliberately
omits a title and stays focused on the curve, clock, and scorecard.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from PIL import Image  # noqa: E402

RECORDS_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECORDS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from make_record_report import (  # noqa: E402
    BG,
    GRID,
    INK,
    MUTED,
    SEED_COLORS,
    TARGET_C,
    aligned_training_series,
    fetch_wandb_history,
    glow,
    parse_train_log,
    set_house_style,
    smooth,
)

DPI = 125
# Animation pacing, as fractions of the animated (pre-hold) timeline.
DRAW_DONE = 0.85   # curve + stopwatch finish here; the tail is dwell on the verdict
PASS_START = 0.50  # pass@1 begins counting up here, overlapping the climb


def hms(seconds: float) -> str:
    s = int(seconds)
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}"


def ease_out(t: float) -> float:
    """Cubic ease-out: fast then settling — reads as deceleration into the final value."""
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t: float) -> float:
    """Smoothstep: slow-fast-slow — a steady, readable draw like a replay scrub."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def record_primary_paths(record_dir: Path) -> tuple[Path, Path] | None:
    """The (train log, eval JSON) the hero reads for a record dir, or None if incomplete.

    Mirrors load_record's selection: first non-sidecar train log, first eval JSON.
    """
    logs = sorted(p for p in record_dir.glob("train_log_seed*.txt") if not p.name.endswith(".flops.txt"))
    evals = sorted(record_dir.glob("eval_seed*.json"))
    return (logs[0], evals[0]) if logs and evals else None


def select_current_record(root: Path, target: float) -> Path:
    """The standing record = fastest wall-clock run whose held-out eval clears the target.

    Tracks the leaderboard's #1 row without hardcoding a dir, so regenerating the hero
    after a new submission is a no-arg command. A run clears the target when its lower
    95% CI (`ci_low`) exceeds it — the same gate as the rules and verify_record.py.
    """
    if not root.is_dir():
        raise SystemExit(f"no records dir at {root}")
    candidates: list[tuple[float, Path]] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        paths = record_primary_paths(d)
        if paths is None:
            continue
        log_path, eval_path = paths
        try:
            final_time = parse_train_log(log_path)["final_time_s"]
            ci_low = json.loads(eval_path.read_text()).get("ci_low")
        except (ValueError, json.JSONDecodeError, OSError):
            continue
        if ci_low is not None and final_time is not None and ci_low > target:
            candidates.append((float(final_time), d))
    if not candidates:
        raise SystemExit(f"no record under {root} clears target {target:.2f} by lower CI")
    return min(candidates)[1]


def load_record(record_dir: Path) -> tuple[dict, dict]:
    paths = record_primary_paths(record_dir)
    if paths is None:
        raise SystemExit(f"{record_dir}: need train_log_seed*.txt and eval_seed*.json")
    log_path, eval_path = paths
    return parse_train_log(log_path), json.loads(eval_path.read_text())


def frame_state(frac: float, log: dict, eval_result: dict, base_pass: float,
                online_frame: object | None = None) -> dict:
    """Map an animated-timeline fraction in [0, 1] to what the frame should show."""
    series = aligned_training_series(log, online_frame)
    final_time = log["final_time_s"]
    final_pass = eval_result["pass_at_1"]

    draw_p = ease_in_out(frac / DRAW_DONE) if frac < DRAW_DONE else 1.0
    n = max(2, round(draw_p * len(series["raw"])))
    record_time = final_time if draw_p >= 1.0 else float(series["xclock"].iloc[n - 1] * 60.0)

    if frac >= PASS_START:
        rp = ease_out((frac - PASS_START) / (1.0 - PASS_START))
        pass_val = base_pass + (final_pass - base_pass) * rp
    else:
        pass_val = None
    return {"n": n, "record_time_s": record_time, "pass_val": pass_val}


def draw_frame(state: dict, log: dict, target: float, out_png: Path | None = None,
               online_frame: object | None = None):
    series = aligned_training_series(log, online_frame)
    x = series["xclock"]
    raw_y = series["raw"]
    y = smooth(raw_y)
    n = state["n"]
    color = SEED_COLORS[0]

    fig = plt.figure(figsize=(11.0, 5.7), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.52, 1.0], left=0.075, right=0.965,
                          top=0.865, bottom=0.135, wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    card = fig.add_subplot(gs[0, 1])
    card.axis("off")

    ax.plot(x.iloc[:n], raw_y.iloc[:n], color=color, alpha=0.13, lw=1.2)
    line = ax.plot(x.iloc[:n], y.iloc[:n], color=color, lw=3.0)[0]
    glow(line, lw=9, alpha=0.28)
    ax.fill_between(x.iloc[:n], y.iloc[:n], color=color, alpha=0.13)
    ax.scatter([x.iloc[n - 1]], [y.iloc[n - 1]], s=62, color=color,
               edgecolor=BG, linewidth=1.6, zorder=5)

    ax.set_xlim(0, max(90, float(x.max()) * 1.03))
    # Headroom from the full raw series (not the drawn slice) so the axis never jumps
    # between frames and the faint ghost spikes don't clip the top edge.
    ax.set_ylim(0.32, max(0.80, float(raw_y.max()) * 1.04))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("record clock (minutes)")
    ax.set_ylabel(series["label"])
    ax.grid(True, axis="y", color=GRID, alpha=0.55)

    card.text(0.0, 0.86, "RECORD", color=MUTED, fontsize=15,
              fontweight="bold", va="top")
    card.text(0.0, 0.755, hms(state["record_time_s"]), color=INK, fontsize=44,
              fontweight="bold", va="top")

    if state["pass_val"] is not None:
        card.text(0.0, 0.45, "HELD-OUT PASS@1", color=MUTED, fontsize=15,
                  fontweight="bold", va="top")
        card.text(0.0, 0.345, f"{state['pass_val']:.1%}", color=color, fontsize=66,
                  fontweight="bold", va="top")
        card.text(0.0, 0.10, f"lower CI > {target:.0%}", color=TARGET_C, fontsize=19,
                  fontweight="bold", va="top")

    if out_png is not None:
        fig.savefig(out_png, dpi=DPI)
    return fig


def render_rgb(state: dict, log: dict, target: float,
               online_frame: object | None = None) -> Image.Image:
    fig = draw_frame(state, log, target, online_frame=online_frame)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_dir", nargs="?", type=Path, default=None,
                        help="record dir to animate; default = current record "
                             "(fastest clearing run under records/)")
    parser.add_argument("--records-root", type=Path, default=RECORDS_DIR,
                        help="where to look for the current record when record_dir is omitted")
    parser.add_argument("--out", type=Path, default=RECORDS_DIR / "hero.gif")
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument("--wandb-runs", default=None,
                        help="optional W&B run path/URL for the train solve-rate curve")
    parser.add_argument("--base-pass", type=float, default=0.57,
                        help="base-model pass@1 the scorecard counts up from")
    parser.add_argument("--frames", type=int, default=66, help="animated frames before the end-hold")
    parser.add_argument("--hold", type=int, default=16, help="duplicate final frames (the dwell before looping)")
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--still", type=Path, default=None,
                        help="instead of the GIF, save one still PNG at --still-frac")
    parser.add_argument("--still-frac", type=float, default=1.0)
    args = parser.parse_args()

    set_house_style()
    record_dir = args.record_dir or select_current_record(args.records_root, args.target)
    if args.record_dir is None:
        print(f"current record: {record_dir}")
    log, eval_result = load_record(record_dir)
    online_frame = fetch_wandb_history([args.wandb_runs])[0] if args.wandb_runs else None
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.still is not None:
        state = frame_state(args.still_frac, log, eval_result, args.base_pass, online_frame)
        draw_frame(state, log, args.target, out_png=args.still, online_frame=online_frame)
        plt.close("all")
        print(f"wrote {args.still}")
        return

    rgb_frames: list[Image.Image] = []
    for frame in range(args.frames):
        frac = (frame + 1) / args.frames
        state = frame_state(frac, log, eval_result, args.base_pass, online_frame)
        rgb_frames.append(render_rgb(state, log, args.target, online_frame))
    rgb_frames += [rgb_frames[-1]] * args.hold  # dwell on the verdict before the loop restarts

    # Quantize every frame against ONE palette built from the final (richest) frame.
    # Per-frame adaptive palettes shimmer on the flat dark background; a shared palette
    # with no dithering keeps the loop clean and the file small.
    pal_src = rgb_frames[-1].convert("P", palette=Image.ADAPTIVE, colors=256)
    frames_p = [im.quantize(palette=pal_src, dither=Image.Dither.NONE) for im in rgb_frames]

    base_ms = round(1000 / args.fps)
    durations = [base_ms] * len(frames_p)
    durations[-1] = 900  # longer final beat so the numbers are readable before it loops

    frames_p[0].save(
        args.out,
        save_all=True,
        append_images=frames_p[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )

    plt.close("all")
    print(f"wrote {args.out}  ({len(frames_p)} frames, {args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
