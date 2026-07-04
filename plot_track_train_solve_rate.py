#!/usr/bin/env python3
"""Per-track train-solve-rate overlay: every record's training curve, both seeds.

One figure per track, single panel: train solve rate vs record clock (the
speedrun axis). Solve rate is the one metric with the same meaning in both
tracks' logs (the LLM track's shaped reward_mean and the non-LLM track's
boxes-on-target proxy are not comparable). A record plots the unfiltered
online solve rate when every seed ships a usable metrics_seed<seed>.jsonl
covering its log (all-or-nothing, so one curve never mixes metrics); older or
partial records fall back to the log's solved_frac, which on the LLM track is
the filtered accepted-batch diagnostic (post-dynamic-sampling) — mixed eras
are marked with a legend dagger. The non-LLM log's solved_frac is already the
true unfiltered episode solve rate, so its fallback is lossless.

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

import itertools
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from make_record_report import (  # noqa: E402
    ONLINE_METRIC, REPO_ROOT, SMOOTH, load_metrics_frame, parse_leaderboard, parse_train_log,
    pct_axis, set_leaderboard_style, train_log_paths,
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
        "ylabel": "train solve rate",
        # This track's step:-line fallback is a DIFFERENT metric (the filtered
        # accepted-batch diagnostic), so all-fallback plots relabel the axis and
        # mixed eras dagger the fallback records.
        "fallback_is_filtered": True,
        "ylabel_fallback": "train solve rate (accepted batches)",
        "out": REPO_ROOT / "assets" / "llm_train_solve_rate.png",
    },
    "non-llm": {
        "lb_section": "Non-LLM Track",
        "records_dir": REPO_ROOT / "non_llm" / "records",
        "title": "Non-LLM track  ·  train solve rate",
        # same metric either way (every episode counts) — no dagger era for this track
        "ylabel": "train solve rate (episodes)",
        "fallback_is_filtered": False,
        "out": REPO_ROOT / "assets" / "non_llm_train_solve_rate.png",
    },
}


def leaderboard_rows(section: str) -> dict[str, dict]:
    """Map record-dir name -> leaderboard row (num, clock, desc) for the legend."""
    return {r["dir"]: r for r in parse_leaderboard(section) if r["dir"]}


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

    All-or-nothing per record (mirroring load_metrics_frames in the report):
    solved_frac is replaced with the unfiltered online rate only when EVERY seed
    ships a usable metrics_seed<seed>.jsonl covering its log — a per-seed or
    per-step splice would mix two different metrics in one curve/band on the LLM
    track. Returns (frames, online)."""
    frames, rates = {}, {}
    for p in train_log_paths(record_dir) + train_log_paths(record_dir / "verification"):
        seed = p.stem.replace("train_log_", "")
        df = parse_train_log(p)["df"]
        w = load_metrics_frame(p.with_name(f"metrics_{seed}.jsonl"), df)
        if w is not None:
            # full coverage is guaranteed by load_metrics_frame — no NaN to fill
            rates[seed] = df.step.map(w.set_index("step")[ONLINE_METRIC])
        frames[seed] = df
    online = bool(frames) and set(rates) == set(frames)
    if online:
        frames = {seed: df.assign(solved_frac=rates[seed]) for seed, df in frames.items()}
    return frames, online


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
        if not frames:
            print(f"  ! {d.name}: no seed logs — skipped")
            continue
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
    any_online = any(r["online"] for r in records)
    mixed = any_online and not all(r["online"] for r in records)
    dagger = mixed and cfg["fallback_is_filtered"]
    legend_title = "line = 2-seed mean  ·  band = seed spread"
    if dagger:
        legend_title += "\n† train metric = filtered accepted batches (pre-metrics record)"

    if len(records) > len(RECORD_COLORS):
        print(f"  ! {len(records)} records exceed the {len(RECORD_COLORS)}-color palette — "
              "colors repeat; extend RECORD_COLORS")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for rec, color in zip(records, itertools.cycle(RECORD_COLORS)):
        b = rec["band"]
        mark = "†" if dagger and not rec["online"] else ""
        ax.fill_between(b.clock_min, b.lo, b.hi, color=color, alpha=0.18, lw=0)
        ax.plot(b.clock_min, b["mean"], color=color, lw=2.4,
                label=f"#{rec['num']}{mark} · {shorten(rec['desc'])} · {rec['clock']}")
    ax.set_title(cfg["title"], loc="left")
    ax.set_xlabel("record clock (minutes)")
    ax.set_ylabel(cfg["ylabel_fallback"] if cfg["fallback_is_filtered"] and not any_online
                  else cfg["ylabel"])
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
