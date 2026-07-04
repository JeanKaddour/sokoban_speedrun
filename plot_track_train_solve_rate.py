#!/usr/bin/env python3
"""Per-track train-solve-rate overlay: every record's training curve, both seeds.

One figure per track, single panel: train solve rate vs record clock (the
speedrun axis). Solve rate is the one metric with the same meaning in both
tracks' logs (the LLM track's shaped reward_mean and the non-LLM track's
boxes-on-target proxy are not comparable). Per seed, the series is the
unfiltered online solve rate from metrics_seed<seed>.jsonl when the record
ships it; older records fall back to the log's solved_frac, which on the LLM
track is the filtered accepted-batch diagnostic (post-dynamic-sampling) —
mixed eras are marked with a legend dagger. The non-LLM log's solved_frac is
already the true unfiltered episode solve rate, so its fallback is lossless.

Each record is one color; the line is the 2-seed mean of the smoothed solve
rate and the shaded band spans the two seeds (submission run + verification
rerun) — with two seeds the band IS the observed seed spread, not a fitted
CI. Seeds are aligned on the training step (the axis where they are exactly
comparable) and the per-step clock is averaged over the two nodes, so node
speed differences don't smear the comparison.

Styled like the leaderboard figures (light, academic), not the dark record
dashboards. Reads only artifacts already in the record dirs
(train_log_seed*.txt and metrics_seed*.jsonl at the record root and under
verification/) plus the README leaderboard rows for the canonical record
number / description / time. Writes:
  assets/llm_train_solve_rate.png
  assets/non_llm_train_solve_rate.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from make_record_report import (  # noqa: E402
    ONLINE_METRIC, REPO_ROOT, SMOOTH, normalize_online_frame, parse_train_log, pct_axis,
    set_leaderboard_style, train_log_paths,
)

# One color per RECORD, in leaderboard order — Okabe-Ito (the standard
# colorblind-safe academic palette) with the orange darkened for white-surface
# contrast. Validated: lightness band, chroma, adjacent-pair CVD separation
# (worst ΔE 37.2 ≥ 12), and contrast ≥ 3:1 all pass on white.
RECORD_COLORS = ["#0072B2", "#B87700", "#009E73", "#D55E00", "#CC79A7"]
LEGEND_DESC_MAX = 50

PLOTS = {
    "llm": {
        "lb_section": "LLM Track",
        "records_dir": REPO_ROOT / "llm" / "records",
        "title": "LLM track  ·  train solve rate",
        # fallback label: records predating metrics_seed*.jsonl only log the filtered
        # accepted-batch diagnostic; online label: the unbiased pre-filter solve rate
        "ylabel": "train solve rate (accepted batches)",
        "ylabel_online": "train solve rate",
        "out": REPO_ROOT / "assets" / "llm_train_solve_rate.png",
    },
    "non-llm": {
        "lb_section": "Non-LLM Track",
        "records_dir": REPO_ROOT / "non_llm" / "records",
        "title": "Non-LLM track  ·  train solve rate",
        # same metric either way (every episode counts) — no dagger era for this track
        "ylabel": "train solve rate (episodes)",
        "ylabel_online": "train solve rate (episodes)",
        "out": REPO_ROOT / "assets" / "non_llm_train_solve_rate.png",
    },
}


def leaderboard_rows(section: str) -> dict[str, dict]:
    """Map record-dir name -> {num, clock, desc} from the README's track table.

    parse_leaderboard() in make_record_report keeps only num/minutes/acc; here the
    legend also needs the human-authored Description and the dir-name link target.
    """
    text = (REPO_ROOT / "README.md").read_text()
    block = next(s for s in re.split(r"^## ", text, flags=re.M) if s.startswith(section))
    rows = {}
    for line in block.splitlines():
        if not re.match(r"\|\s*\d+\s*\|", line):  # a data row: leading | <int> |
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        link = re.search(r"\(((?:llm|non_llm)/records/[^)]+?)/?\)", cells[4])
        if not link:
            continue
        rows[Path(link.group(1)).name] = {
            "num": int(cells[0]), "clock": cells[1],
            "desc": cells[2].replace("`", ""),
        }
    return rows


def shorten(desc: str, limit: int = LEGEND_DESC_MAX) -> str:
    """Trim a leaderboard description to legend width at comma boundaries."""
    if len(desc) <= limit:
        return desc
    parts = desc.split(", ")
    keep = parts[:1]
    for p in parts[1:]:
        if len(", ".join(keep + [p])) > limit:
            break
        keep.append(p)
    out = ", ".join(keep)
    return out if len(out) <= limit else out[:limit - 1] + "…"


def load_seed_frames(record_dir: Path) -> tuple[dict[str, pd.DataFrame], bool]:
    """Both seeds' step series: the submission log + the verification rerun's.

    When a seed ships metrics_seed<seed>.jsonl (records assembled after speedrun.py
    grew the metrics stream), its solved_frac column is replaced with the unfiltered
    online solve rate; otherwise the log's value stands. Returns (frames, online) —
    online is True only if every seed came from a metrics file."""
    frames, online = {}, []
    for p in train_log_paths(record_dir) + train_log_paths(record_dir / "verification"):
        seed = p.stem.replace("train_log_", "")
        df = parse_train_log(p)["df"]
        metrics_path = p.with_name(f"metrics_{seed}.jsonl")
        if metrics_path.exists():
            w = normalize_online_frame(pd.read_json(metrics_path, lines=True))
            rate = df.step.map(w.set_index("step")[ONLINE_METRIC])
            df = df.assign(solved_frac=rate.fillna(df.solved_frac))
        online.append(metrics_path.exists())
        frames[seed] = df
    return frames, bool(online) and all(online)


def seed_band(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Step-aligned 2-seed summary: smoothed solve rate mean/min/max + mean clock."""
    rates = pd.concat(
        [f.set_index("step").solved_frac.rename(seed) for seed, f in frames.items()],
        axis=1, join="inner")
    clocks = pd.concat(
        [f.set_index("step").record_time_s.rename(seed) for seed, f in frames.items()],
        axis=1, join="inner")
    win = max(SMOOTH, len(rates) // 40)  # ~8 for 50-75-step LLM runs, ~30 for 1k+-step non-LLM
    sm = rates.rolling(win, min_periods=1).mean()
    return pd.DataFrame({
        "mean": sm.mean(axis=1), "lo": sm.min(axis=1), "hi": sm.max(axis=1),
        "clock_min": clocks.mean(axis=1) / 60.0,
    }).reset_index(drop=True)


def plot_track(cfg: dict) -> None:
    rows = leaderboard_rows(cfg["lb_section"])
    records = []
    for d in sorted(cfg["records_dir"].iterdir()):
        if not (d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_", d.name)):
            continue
        frames, online = load_seed_frames(d)
        if len(frames) < 2:
            print(f"  ! {d.name}: only {len(frames)} seed log(s) — band collapses to one seed")
        lb = rows.get(d.name, {"num": len(records) + 1, "clock": "?", "desc": d.name})
        records.append({"name": d.name, "band": seed_band(frames), "seeds": sorted(frames),
                        "online": online, **lb})
    if not records:
        raise SystemExit(f"no record dirs under {cfg['records_dir']}")
    records.sort(key=lambda r: r["num"])

    # All-online records get the plain label; a mixed era daggers the fallback records
    # (only meaningful where fallback is a different metric, i.e. the LLM track).
    all_online = all(r["online"] for r in records)
    mixed = not all_online and any(r["online"] for r in records)
    dagger = mixed and cfg["ylabel"] != cfg["ylabel_online"]
    legend_title = "line = 2-seed mean  ·  band = seed spread"
    if dagger:
        legend_title += "\n† train metric = filtered accepted batches (pre-metrics record)"

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for rec, color in zip(records, RECORD_COLORS):
        b = rec["band"]
        mark = "†" if dagger and not rec["online"] else ""
        ax.fill_between(b.clock_min, b.lo, b.hi, color=color, alpha=0.18, lw=0)
        ax.plot(b.clock_min, b["mean"], color=color, lw=2.4,
                label=f"#{rec['num']}{mark} · {shorten(rec['desc'])} · {rec['clock']}")
    ax.set_title(cfg["title"], loc="left")
    ax.set_xlabel("record clock (minutes)")
    ax.set_ylabel(cfg["ylabel_online"] if all_online or mixed else cfg["ylabel"])
    ax.set_xlim(left=0)
    pct_axis(ax)
    leg = ax.legend(loc="lower right", fontsize=11,
                    title=legend_title, title_fontsize=10.5)
    leg.get_title().set_style("italic")
    leg.get_title().set_color("#333a42")
    fig.savefig(cfg["out"])
    plt.close(fig)
    out = cfg["out"]
    out_name = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
    print(f"wrote {out_name}  "
          f"({len(records)} records: " + ", ".join(f"#{r['num']} {r['seeds']}" for r in records) + ")")


def main() -> None:
    set_leaderboard_style()
    for cfg in PLOTS.values():
        plot_track(cfg)


if __name__ == "__main__":
    main()
