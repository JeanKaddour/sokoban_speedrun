#!/usr/bin/env python3
"""Per-track training-metric overlays: every record's training curve, both seeds.

By default this preserves the original train-solve-rate figure. Pass --metric
to plot another canonical record metric, or --all-record-metrics to generate
the common record overlays:

  python plot_track_train_solve_rate.py
  python plot_track_train_solve_rate.py --metric loss --metric grad_norm
  python plot_track_train_solve_rate.py --all-record-metrics

Each figure is one metric vs record clock (the speedrun axis). A record uses
metrics_seed<seed>.jsonl only when every seed ships a usable file covering its
log (all-or-nothing, so one curve never mixes metric sources). If the selected
metric has a step-log fallback, older or partial records use that parsed log
column. For the LLM track's online solve-rate plot, that fallback is the
filtered accepted-batch diagnostic (post-dynamic-sampling), so mixed eras are
marked with a legend dagger. The non-LLM log's solved_frac is already the true
unfiltered episode solve rate, so its fallback is lossless.

Each record is one color; the line is the 2-seed mean of the smoothed metric
value and the shaded band spans the two seeds (submission run + verification
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

import argparse
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from make_record_report import (  # noqa: E402
    ONLINE_METRIC, REPO_ROOT, SMOOTH, parse_leaderboard, parse_train_log,
    pct_axis, set_leaderboard_style, train_log_paths,
)

# One color per RECORD, in leaderboard order — Okabe-Ito (the standard
# colorblind-safe academic palette) with the orange darkened for white-surface
# contrast. Validated: lightness band, chroma, adjacent-pair CVD separation
# (worst ΔE 37.2 ≥ 12), and contrast ≥ 3:1 all pass on white.
RECORD_COLORS = ["#0072B2", "#B87700", "#009E73", "#D55E00", "#CC79A7"]
LEGEND_DESC_MAX = 50


@dataclass(frozen=True)
class MetricSpec:
    key: str
    slug: str
    title: str
    ylabel: str
    fallback_col: str | None = None
    percent: bool = False
    yscale: str = "linear"
    fallback_ylabel: str | None = None
    dagger_filtered_fallback: bool = False


METRIC_SPECS = {
    "train_solve_rate": MetricSpec(
        key=ONLINE_METRIC,
        slug="train_solve_rate",
        title="train solve rate",
        ylabel="train solve rate",
        fallback_col="solved_frac",
        percent=True,
        fallback_ylabel="train solve rate (accepted batches)",
        dagger_filtered_fallback=True,
    ),
    "solved_frac": MetricSpec(
        key="record/solved_frac",
        slug="solved_frac",
        title="batch solve fraction",
        ylabel="batch solve fraction",
        fallback_col="solved_frac",
        percent=True,
    ),
    "reward_mean": MetricSpec(
        key="record/reward_mean",
        slug="reward_mean",
        title="reward mean",
        ylabel="reward mean",
        fallback_col="reward_mean",
    ),
    "loss": MetricSpec(
        key="record/loss",
        slug="loss",
        title="loss",
        ylabel="loss",
        fallback_col="loss",
    ),
    "grad_norm": MetricSpec(
        key="record/grad_norm",
        slug="grad_norm",
        title="gradient norm",
        ylabel="gradient norm",
        fallback_col="grad_norm",
        yscale="log",
    ),
}
METRIC_ALIASES = {
    **{name: spec for name, spec in METRIC_SPECS.items()},
    **{spec.key: spec for spec in METRIC_SPECS.values()},
}
DEFAULT_METRICS = ("train_solve_rate",)
ALL_RECORD_METRICS = ("train_solve_rate", "solved_frac", "reward_mean", "loss", "grad_norm")

PLOTS = {
    "llm": {
        "lb_section": "LLM Track",
        "records_dir": REPO_ROOT / "llm" / "records",
        "title_prefix": "LLM track",
        # This track's step:-line fallback is a DIFFERENT metric (the filtered
        # accepted-batch diagnostic), so all-fallback plots relabel the axis and
        # mixed eras dagger the fallback records.
        "fallback_is_filtered": True,
    },
    "non-llm": {
        "lb_section": "Non-LLM Track",
        "records_dir": REPO_ROOT / "non_llm" / "records",
        "title_prefix": "Non-LLM track",
        # same metric either way (every episode counts) — no dagger era for this track
        "fallback_is_filtered": False,
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


def slugify_metric(metric: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_")
    return slug or "metric"


def resolve_metric(name: str) -> MetricSpec:
    if name in METRIC_ALIASES:
        return METRIC_ALIASES[name]
    return MetricSpec(
        key=name,
        slug=slugify_metric(name),
        title=name,
        ylabel=name,
    )


def output_path(track: str, metric: MetricSpec, out_dir: Path) -> Path:
    return out_dir / f"{track.replace('-', '_')}_{metric.slug}.png"


def load_metric_frame(path: Path, log_df: pd.DataFrame, metric: MetricSpec) -> pd.DataFrame | None:
    """One seed's metrics_seed<seed>.jsonl metric series, or None for fallback."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        raw = pd.read_json(path, lines=True)
    except ValueError as exc:
        print(f"{path.name}: unusable metrics file ({exc}); falling back to step: lines")
        return None
    if "step" not in raw.columns or metric.key not in raw.columns:
        return None
    out = raw[["step", metric.key]].dropna().rename(columns={metric.key: "value"}).copy()
    try:
        out["step"] = pd.to_numeric(out["step"], errors="raise").astype(int)
        out["value"] = pd.to_numeric(out["value"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        print(f"{path.name}: unusable {metric.key} values ({exc}); falling back to step: lines")
        return None
    out = out.sort_values("step", kind="stable").drop_duplicates("step", keep="last")
    out = out[out["step"].isin(log_df.step)].reset_index(drop=True)
    if not log_df.step.isin(out["step"]).all():
        print(f"{path.name}: {metric.key} covers {len(out)}/{len(log_df)} log steps; "
              "falling back to step: lines")
        return None
    return out


def log_fallback_frame(df: pd.DataFrame, metric: MetricSpec) -> pd.DataFrame | None:
    if metric.fallback_col is None or metric.fallback_col not in df.columns:
        return None
    return df[["step", "record_time_s", metric.fallback_col]].rename(
        columns={metric.fallback_col: "value"}
    ).dropna().copy()


def load_seed_frames(record_dir: Path, metric: MetricSpec) -> tuple[dict[str, pd.DataFrame], str]:
    """Both seeds' step series: the submission log + the verification rerun's.

    All-or-nothing per record: metrics_seed<seed>.jsonl is used only when EVERY
    seed has the requested metric covering its log. Otherwise the parsed step-log
    fallback is used when the metric defines one. Returns (frames, source), where
    source is "metrics", "fallback", or "missing"."""
    metric_frames, fallback_frames, seeds = {}, {}, []
    for p in train_log_paths(record_dir) + train_log_paths(record_dir / "verification"):
        seed = p.stem.replace("train_log_", "")
        seeds.append(seed)
        df = parse_train_log(p)["df"]
        w = load_metric_frame(p.with_name(f"metrics_{seed}.jsonl"), df, metric)
        if w is not None:
            # full coverage is guaranteed by load_metric_frame — no NaN to fill
            metric_frames[seed] = df[["step", "record_time_s"]].assign(
                value=df.step.map(w.set_index("step")["value"])
            )
        fallback = log_fallback_frame(df, metric)
        if fallback is not None:
            fallback_frames[seed] = fallback
    if seeds and set(metric_frames) == set(seeds):
        return metric_frames, "metrics"
    if seeds and set(fallback_frames) == set(seeds):
        return fallback_frames, "fallback"
    return {}, "missing"


def seed_band(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Step-aligned 2-seed summary: smoothed metric mean/min/max + mean clock."""
    values = pd.concat(
        [f.set_index("step").value.rename(seed) for seed, f in frames.items()],
        axis=1, join="inner")
    clocks = pd.concat(
        [f.set_index("step").record_time_s.rename(seed) for seed, f in frames.items()],
        axis=1, join="inner")
    win = max(SMOOTH, len(values) // 40)  # ~8 for 50-75-step LLM runs, ~30 for 1k+-step non-LLM
    sm = values.rolling(win, min_periods=1).mean()
    return pd.DataFrame({
        "mean": sm.mean(axis=1), "lo": sm.min(axis=1), "hi": sm.max(axis=1),
        "clock_min": clocks.mean(axis=1) / 60.0,
    }).reset_index(drop=True)


def plot_track(track: str, cfg: dict, metric: MetricSpec, out_dir: Path) -> None:
    rows = leaderboard_rows(cfg["lb_section"])
    records = []
    for d in sorted(cfg["records_dir"].iterdir()):
        if not (d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_", d.name)):
            continue
        frames, source = load_seed_frames(d, metric)
        if not frames:
            print(f"  ! {d.name}: no usable {metric.key} series — skipped")
            continue
        if len(frames) < 2:
            print(f"  ! {d.name}: only {len(frames)} seed log(s) — band collapses to one seed")
        lb = rows.get(d.name, {"num": len(records) + 1, "clock": "?", "desc": d.name})
        records.append({"name": d.name, "band": seed_band(frames), "seeds": sorted(frames),
                        "source": source, **lb})
    if not records:
        raise SystemExit(f"no record dirs with usable {metric.key} under {cfg['records_dir']}")
    records.sort(key=lambda r: r["num"])

    # All-metrics records get the plain label; a mixed era daggers the fallback records
    # (only meaningful where fallback is a different metric, i.e. the LLM track).
    any_metrics = any(r["source"] == "metrics" for r in records)
    mixed = any_metrics and not all(r["source"] == "metrics" for r in records)
    dagger = mixed and metric.dagger_filtered_fallback and cfg["fallback_is_filtered"]
    legend_title = "line = 2-seed mean  ·  band = seed spread"
    if dagger:
        legend_title += "\n† train metric = filtered accepted batches (pre-metrics record)"

    if len(records) > len(RECORD_COLORS):
        print(f"  ! {len(records)} records exceed the {len(RECORD_COLORS)}-color palette — "
              "colors repeat; extend RECORD_COLORS")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for rec, color in zip(records, itertools.cycle(RECORD_COLORS)):
        b = rec["band"]
        mark = "†" if dagger and rec["source"] != "metrics" else ""
        ax.fill_between(b.clock_min, b.lo, b.hi, color=color, alpha=0.18, lw=0)
        ax.plot(b.clock_min, b["mean"], color=color, lw=2.4,
                label=f"#{rec['num']}{mark} · {shorten(rec['desc'])} · {rec['clock']}")
    ax.set_title(f"{cfg['title_prefix']}  ·  {metric.title}", loc="left")
    ax.set_xlabel("record clock (minutes)")
    ylabel = metric.ylabel
    if metric.slug == "train_solve_rate" and track == "non-llm":
        ylabel = "train solve rate (episodes)"
    if (metric.fallback_ylabel and cfg["fallback_is_filtered"] and not any_metrics):
        ylabel = metric.fallback_ylabel
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    if metric.percent:
        pct_axis(ax)
    if metric.yscale != "linear":
        ax.set_yscale(metric.yscale)
    leg = ax.legend(loc="lower right", fontsize=11,
                    title=legend_title, title_fontsize=10.5)
    leg.get_title().set_style("italic")
    leg.get_title().set_color("#333a42")
    out = output_path(track, metric, out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    out_name = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
    print(f"wrote {out_name}  "
          f"({metric.key}; {len(records)} records: "
          + ", ".join(f"#{r['num']} {r['seeds']}" for r in records) + ")")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot per-track record overlays for training metrics."
    )
    ap.add_argument(
        "--track",
        choices=("all", *PLOTS.keys()),
        default="all",
        help="track to plot (default: all)",
    )
    ap.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        metavar="NAME_OR_COLUMN",
        help=(
            "metric to plot; presets: "
            + ", ".join(METRIC_SPECS)
            + ". May also be a raw metrics.jsonl column."
        ),
    )
    ap.add_argument(
        "--all-record-metrics",
        action="store_true",
        help="plot train_solve_rate, solved_frac, reward_mean, loss, and grad_norm",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "assets",
        help="output directory (default: assets/)",
    )
    return ap.parse_args()


def selected_metrics(args: argparse.Namespace) -> list[MetricSpec]:
    names = list(ALL_RECORD_METRICS if args.all_record_metrics else (args.metrics or DEFAULT_METRICS))
    out, seen = [], set()
    for name in names:
        metric = resolve_metric(name)
        if metric.key in seen:
            continue
        out.append(metric)
        seen.add(metric.key)
    return out


def main() -> None:
    args = parse_args()
    tracks = PLOTS if args.track == "all" else {args.track: PLOTS[args.track]}
    metrics = selected_metrics(args)
    set_leaderboard_style()
    for metric in metrics:
        for track, cfg in tracks.items():
            plot_track(track, cfg, metric, args.out_dir)


if __name__ == "__main__":
    main()
