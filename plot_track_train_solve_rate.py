#!/usr/bin/env python3
"""Per-track training-metric overlays: record training curves across available seeds.

By default this preserves the original train-solve-rate figure. ``--recent``
is the submission-pipeline mode: it plots the latest three records, chooses a
solve metric that is comparable across the whole window, and writes the stable
README asset names. Pass --metric to plot another canonical record metric, or
--all-record-metrics to generate the common record overlays:

  python plot_track_train_solve_rate.py
  python plot_track_train_solve_rate.py --recent
  python plot_track_train_solve_rate.py --track llm --latest 3
  python plot_track_train_solve_rate.py --track llm --latest 3 --metric solved_frac
  python plot_track_train_solve_rate.py --metric loss --metric grad_norm
  python plot_track_train_solve_rate.py --all-record-metrics

Each figure is one metric vs record clock (the speedrun axis). A record uses
metrics_seed<seed>.jsonl only when every seed ships a usable file covering its
log (all-or-nothing, so one curve never mixes metric sources). Newer LLM text
logs also carry the same unfiltered online solve-rate inline; that exact bridge
is preferred when a record predates metrics.jsonl. If the selected metric has a
step-log fallback, older records use that parsed log column. For the LLM track's
online solve-rate plot, the oldest fallback is the filtered accepted-batch
diagnostic (post-dynamic-sampling), so mixed eras are marked with a legend
dagger. The non-LLM log's solved_frac is already the true unfiltered episode
solve rate, so its fallback is lossless.

For a rolling LLM figure that spans legacy records without the exact online
metric, ``--recent`` uses solved_frac to keep the accepted-batch metric
consistent across every selected record. Once all recent records ship inline
online_solved or metrics JSONL, it automatically switches the rolling window to
the exact unfiltered train solve rate. The non-LLM fallback is already the exact
episode solve rate.

Each record is one color; the line is the mean of the smoothed metric value and
the shaded band spans the available seeds. For a verified record these are the
submission run + verification rerun, so the band is the observed two-seed
spread, not a fitted CI. Before verification, the new record temporarily has a
collapsed single-seed band. Seeds are aligned on the training step (the axis
where they are exactly comparable) and the per-step clock is averaged over the
nodes, so node speed differences don't smear the comparison.

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
LEGEND_DESC_MAX = 44


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
RECENT_RECORD_COUNT = 3

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


def inline_metric_frame(df: pd.DataFrame, metric: MetricSpec) -> pd.DataFrame | None:
    """Exact metric recovered from a newer durable text-log suffix."""
    if metric.key != ONLINE_METRIC or "online_solved_frac" not in df.columns:
        return None
    out = df[["step", "record_time_s", "online_solved_frac"]].rename(
        columns={"online_solved_frac": "value"}
    )
    if out["value"].isna().any():
        return None
    return out.copy()


def load_seed_frames(record_dir: Path, metric: MetricSpec) -> tuple[dict[str, pd.DataFrame], str]:
    """Available step series: the submission log + verification rerun logs.

    All-or-nothing per record: metrics_seed<seed>.jsonl is used only when EVERY
    seed has the requested metric covering its log. Otherwise the parsed step-log
    fallback is used when the metric defines one. Returns (frames, source), where
    source is "metrics", "inline", "fallback", or "missing"."""
    metric_frames, inline_frames, fallback_frames, seeds = {}, {}, {}, []
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
        inline = inline_metric_frame(df, metric)
        if inline is not None:
            inline_frames[seed] = inline
        fallback = log_fallback_frame(df, metric)
        if fallback is not None:
            fallback_frames[seed] = fallback
    if seeds and set(metric_frames) == set(seeds):
        return metric_frames, "metrics"
    if seeds and set(inline_frames) == set(seeds):
        return inline_frames, "inline"
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


def plot_track(track: str, cfg: dict, metric: MetricSpec, out_dir: Path,
               latest: int | None = None, out_path: Path | None = None) -> None:
    rows = leaderboard_rows(cfg["lb_section"])
    records = []
    for d in sorted(cfg["records_dir"].iterdir()):
        if not (d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_", d.name)):
            continue
        if latest is not None and d.name not in rows:
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
    if latest is not None:
        records = records[-latest:]

    # Exact metrics (JSONL or the inline bridge) get the plain label; a mixed era
    # daggers the fallback records (only meaningful where fallback differs).
    exact_sources = {"metrics", "inline"}
    any_exact = any(r["source"] in exact_sources for r in records)
    mixed = any_exact and not all(r["source"] in exact_sources for r in records)
    dagger = mixed and metric.dagger_filtered_fallback and cfg["fallback_is_filtered"]
    if all(len(r["seeds"]) == 2 for r in records):
        legend_title = "line: 2-seed mean  ·  band: seed range"
    else:
        legend_title = "line: available-seed mean  ·  band: seed range"
    if dagger:
        legend_title += "\n† train metric = filtered accepted batches (legacy record)"

    if len(records) > len(RECORD_COLORS):
        print(f"  ! {len(records)} records exceed the {len(RECORD_COLORS)}-color palette — "
              "colors repeat; extend RECORD_COLORS")
    recent = latest is not None
    fig, ax = plt.subplots(
        figsize=(9.6, 5.4) if recent else (8.4, 4.9),
        constrained_layout=recent,
    )
    for rec, color in zip(records, itertools.cycle(RECORD_COLORS)):
        b = rec["band"]
        mark = "†" if dagger and rec["source"] not in exact_sources else ""
        ax.fill_between(b.clock_min, b.lo, b.hi, color=color,
                        alpha=0.22 if recent else 0.18, lw=0)
        desc = shorten(rec["desc"], 34 if recent else LEGEND_DESC_MAX)
        ax.plot(b.clock_min, b["mean"], color=color, lw=3.0 if recent else 2.4,
                label=f"#{rec['num']}{mark} · {desc} · {rec['clock']}")
    title = (
        f"{cfg['title_prefix']} · recent record training curves"
        if latest is not None
        else f"{cfg['title_prefix']} · record history · {metric.title}"
    )
    ax.set_title(
        title,
        loc="left",
        fontsize=17,
        pad=10,
    )
    ax.set_xlabel("record clock (minutes)")
    ylabel = metric.ylabel
    if metric.slug == "train_solve_rate" and track == "non-llm":
        ylabel = "train solve rate (episodes)"
    if (metric.fallback_ylabel and cfg["fallback_is_filtered"] and not any_exact):
        ylabel = metric.fallback_ylabel
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    if metric.percent:
        pct_axis(ax)
    if metric.yscale != "linear":
        ax.set_yscale(metric.yscale)
    if recent:
        leg = ax.legend(
            loc="lower right",
            fontsize=9.5,
            title=legend_title,
            title_fontsize=9.0,
            frameon=True,
            facecolor="white",
            edgecolor="#c7ccd3",
            framealpha=0.94,
        )
    else:
        leg = ax.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.24),
            borderaxespad=0,
            fontsize=10.5,
            title=legend_title,
            title_fontsize=9.5,
        )
    leg.get_title().set_style("italic")
    leg.get_title().set_color("#333a42")
    out = out_path or output_path(track, metric, out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Pin rolling README/X assets to an exact 16:9 canvas. The normal plotting
    # mode keeps its tight crop because it may have an outside legend.
    fig.savefig(out, bbox_inches=fig.bbox_inches if recent else "tight")
    plt.close(fig)
    out_name = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
    print(f"wrote {out_name}  "
          f"({metric.key}; {len(records)} records: "
          + ", ".join(f"#{r['num']} {r['seeds']}" for r in records) + ")")


def recent_metric(track: str, cfg: dict, latest: int) -> MetricSpec:
    """Choose one semantically consistent solve metric for the rolling window.

    On the LLM track, the old step-log fallback for online solve rate is a
    filtered accepted-batch diagnostic rather than the exact unfiltered metric.
    Keep the entire window on solved_frac while any selected record is from that
    legacy era; switch automatically once every selected record has exact JSONL
    or inline online_solved data. Non-LLM solved_frac is losslessly equivalent to
    the online episode solve rate, so its normal train_solve_rate view is safe.
    """
    online = METRIC_SPECS["train_solve_rate"]
    if not cfg["fallback_is_filtered"]:
        return online

    rows = leaderboard_rows(cfg["lb_section"])
    sources = []
    for d in sorted(cfg["records_dir"].iterdir()):
        if not (d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_", d.name)):
            continue
        if d.name not in rows:
            continue
        frames, source = load_seed_frames(d, online)
        if not frames:
            continue
        lb = rows.get(d.name, {"num": len(sources) + 1})
        sources.append((lb["num"], d.name, source))
    sources.sort()
    selected = sources[-latest:]
    if selected and all(source in {"metrics", "inline"} for _num, _name, source in selected):
        print("recent curves: all selected LLM records have exact online solve rate")
        return online

    legacy = ", ".join(f"#{num}" for num, _name, source in selected
                       if source not in {"metrics", "inline"})
    print("recent curves: using record/solved_frac across the LLM window"
          + (f" (legacy online metric: {legacy})" if legacy else ""))
    return METRIC_SPECS["solved_frac"]


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
        "--latest",
        type=int,
        default=None,
        metavar="N",
        help="plot only the latest N leaderboard records (default: all)",
    )
    ap.add_argument(
        "--recent",
        action="store_true",
        help=(
            "submission-pipeline mode: plot the latest 3 records (or --latest N), "
            "auto-select a comparable solve metric, and write stable *_train_solve_rate.png assets"
        ),
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
    if args.latest is not None and args.latest < 1:
        raise SystemExit("--latest must be at least 1")
    tracks = PLOTS if args.track == "all" else {args.track: PLOTS[args.track]}
    if args.recent:
        if args.metrics or args.all_record_metrics:
            raise SystemExit("--recent chooses its metric automatically; do not combine it with --metric/--all-record-metrics")
        latest = args.latest or RECENT_RECORD_COUNT
        set_leaderboard_style()
        for track, cfg in tracks.items():
            metric = recent_metric(track, cfg, latest)
            out = args.out_dir / f"{track.replace('-', '_')}_train_solve_rate.png"
            plot_track(track, cfg, metric, args.out_dir, latest=latest, out_path=out)
        return

    metrics = selected_metrics(args)
    set_leaderboard_style()
    for metric in metrics:
        for track, cfg in tracks.items():
            plot_track(track, cfg, metric, args.out_dir, latest=args.latest)


if __name__ == "__main__":
    main()
