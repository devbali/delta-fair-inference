#!/usr/bin/env python3
"""
Regenerate and aggregate experiment plots directly from their CSV sources.

The script loads latency and token-usage metrics from each provided experiment
directory, mirrors the plotting logic used in the per-experiment `plot.py`
files, and renders combined figures with one panel per traffic pattern.

Usage example:
    python combine_experiment_plots.py EXP1 EXP2 EXP3 EXP4 --tag 10-15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

# Shared configuration -----------------------------------------------------
VARIANT_ORDER = ["1s", "60s", "vtc", "plain"]
CUT_FIRST_SECONDS = 90
CUT_LAST_SECONDS = 300
ROLLING_WINDOW = 5
GOOD_CLIENT_MAX = 20
TOTAL_CLIENT_MAX = 20

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 12
LEGEND_TITLE_FONTSIZE = 13
SUPTITLE_FONTSIZE = 22


@dataclass
class ExperimentData:
    name: str
    variant: str
    latency_df: pd.DataFrame
    throughput_df: pd.DataFrame
    latency_start: pd.Timestamp
    latency_end: pd.Timestamp


# ---------------------------------------------------------------------------
# Helpers for loading and preparing experiment data
# ---------------------------------------------------------------------------

def infer_variant(exp_path: Path) -> str:
    """Infer the traffic variant encoded in the experiment directory name."""
    name = exp_path.name.lower()
    if "-1s-" in name:
        return "1s"
    if "-60s-" in name:
        return "60s"
    if "-vtc-" in name:
        return "vtc"
    if "-plain-" in name:
        return "plain"
    return name  # Fallback to original name if no pattern matches


def collect_experiments(exp_paths: Iterable[Path]) -> Dict[str, Path]:
    """
    Return a mapping from traffic variant to experiment directory.

    Raises:
        FileNotFoundError: if a provided path is missing.
        ValueError: if two paths map to the same variant.
    """
    variant_map: Dict[str, Path] = {}
    for path in exp_paths:
        if not path.exists():
            raise FileNotFoundError(f"Experiment path not found: {path}")
        variant = infer_variant(path)
        if variant in variant_map:
            raise ValueError(
                f"Duplicate variant '{variant}' supplied:\n"
                f"  - {variant_map[variant]}\n"
                f"  - {path}"
            )
        variant_map[variant] = path
    return variant_map


def _pick_latest_csv(exp_path: Path, prefix: str) -> Optional[Path]:
    """Return the newest CSV matching the given prefix inside exp_path."""
    candidates = sorted(exp_path.glob(f"{prefix}_*.csv"))
    return candidates[-1] if candidates else None


def load_experiment(exp_path: Path) -> ExperimentData:
    """Load latency and throughput data for a single experiment directory."""
    lat_path = _pick_latest_csv(exp_path, "latencies")
    if lat_path is None:
        raise FileNotFoundError(f"No latency CSV found in {exp_path}")

    df_lat = pd.read_csv(lat_path)
    df_lat["timestamp_start_iso"] = pd.to_datetime(
        df_lat["timestamp_start_iso"], utc=True
    )
    df_lat = df_lat.sort_values("timestamp_start_iso")

    original_t0 = df_lat["timestamp_start_iso"].iloc[0]
    df_lat["time_since_start_s"] = (
        df_lat["timestamp_start_iso"] - original_t0
    ).dt.total_seconds()

    df_lat = df_lat[
        (df_lat["time_since_start_s"] >= CUT_FIRST_SECONDS)
        & (df_lat["time_since_start_s"] <= CUT_LAST_SECONDS)
    ].copy()
    if df_lat.empty:
        raise ValueError(
            f"No latency data left after applying time bounds "
            f"[{CUT_FIRST_SECONDS}, {CUT_LAST_SECONDS}]s "
            f"in {exp_path}"
        )

    latency_start = df_lat["timestamp_start_iso"].min()
    latency_end = df_lat["timestamp_start_iso"].max()

    df_lat["avg_tbt"] = np.where(
        df_lat["completion_tokens"] > 0,
        df_lat["duration_taft"] / df_lat["completion_tokens"],
        np.nan,
    )
    df_lat["ttft"] = df_lat["duration_ttft"]
    df_lat["kind"] = df_lat["kind"].astype(str)

    throughput_df = compute_throughput_series(
        exp_path,
        original_t0,
        latency_start,
        latency_end,
    )

    return ExperimentData(
        name=exp_path.name,
        variant=infer_variant(exp_path),
        latency_df=df_lat,
        throughput_df=throughput_df,
        latency_start=latency_start,
        latency_end=latency_end,
    )


def compute_throughput_series(
    exp_path: Path,
    t0: pd.Timestamp,
    latency_start: pd.Timestamp,
    latency_end: pd.Timestamp,
) -> pd.DataFrame:
    """Compute per-client-group throughput series for an experiment."""
    token_path = _pick_latest_csv(exp_path, "token_usage")
    if token_path is None:
        return pd.DataFrame()

    token_df = pd.read_csv(token_path)
    if "timestamp_iso" not in token_df.columns:
        return pd.DataFrame()

    token_df["timestamp"] = pd.to_datetime(token_df["timestamp_iso"], utc=True)
    token_df = token_df.sort_values(["user_id", "timestamp"])
    token_df = token_df[token_df["user_id"].between(1, TOTAL_CLIENT_MAX)].copy()

    token_df["elapsed_s"] = (
        token_df.groupby("user_id")["timestamp"].diff().dt.total_seconds()
    )
    token_df.loc[token_df["elapsed_s"] <= 0, "elapsed_s"] = np.nan

    prefill_delta = (
        token_df.groupby("user_id")["prefill_tokens"]
        .diff()
        .clip(lower=0)
    )
    decode_delta = (
        token_df.groupby("user_id")["decode_tokens"]
        .diff()
        .clip(lower=0)
    )
    total_tokens = token_df["prefill_tokens"] + token_df["decode_tokens"]
    total_delta = (
        token_df.assign(total_tokens=total_tokens)
        .groupby("user_id")["total_tokens"]
        .diff()
        .clip(lower=0)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        token_df["prefill_throughput"] = np.where(
            token_df["elapsed_s"] > 0,
            prefill_delta / token_df["elapsed_s"],
            np.nan,
        )
        token_df["decode_throughput"] = np.where(
            token_df["elapsed_s"] > 0,
            decode_delta / token_df["elapsed_s"],
            np.nan,
        )
        token_df["total_throughput"] = np.where(
            token_df["elapsed_s"] > 0,
            total_delta / token_df["elapsed_s"],
            np.nan,
        )

    token_df = token_df[
        (token_df["timestamp"] >= latency_start)
        & (token_df["timestamp"] <= latency_end)
    ].copy()
    if token_df.empty:
        return pd.DataFrame()

    token_df["time_since_start_s"] = (
        token_df["timestamp"] - t0
    ).dt.total_seconds()
    token_df = token_df[
        (token_df["time_since_start_s"] >= CUT_FIRST_SECONDS)
        & (token_df["time_since_start_s"] <= CUT_LAST_SECONDS)
    ]
    if token_df.empty:
        return pd.DataFrame()

    token_df["client_group"] = np.where(
        token_df["user_id"] <= GOOD_CLIENT_MAX,
        f"Good clients (1-{GOOD_CLIENT_MAX})",
        f"Bad clients ({GOOD_CLIENT_MAX + 1}-{TOTAL_CLIENT_MAX})",
    )

    throughput_metrics = [
        "prefill_throughput",
        "decode_throughput",
        "total_throughput",
    ]

    grouped = (
        token_df.dropna(subset=throughput_metrics)
        .groupby(["client_group", "timestamp"])[throughput_metrics]
        .mean()
        .reset_index()
    )
    if grouped.empty:
        return pd.DataFrame()

    grouped["time_since_start_s"] = (
        grouped["timestamp"] - t0
    ).dt.total_seconds()
    grouped = grouped[
        (grouped["time_since_start_s"] >= CUT_FIRST_SECONDS)
        & (grouped["time_since_start_s"] <= CUT_LAST_SECONDS)
    ]
    return grouped.sort_values(["client_group", "time_since_start_s"])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def configure_axis(
    ax: plt.Axes,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)


def add_latency_scatter_panel(fig: plt.Figure, outer_spec, data: ExperimentData) -> None:
    inner_spec: GridSpec = outer_spec.subgridspec(2, 1, hspace=0.05)
    ax_tbt = fig.add_subplot(inner_spec[0, 0])
    ax_ttft = fig.add_subplot(inner_spec[1, 0], sharex=ax_tbt)

    for kind, group in data.latency_df.groupby("kind"):
        ax_tbt.scatter(
            group["time_since_start_s"],
            group["avg_tbt"],
            label=kind,
            s=20,
            alpha=0.55,
        )
        ax_ttft.scatter(
            group["time_since_start_s"],
            group["ttft"],
            label=kind,
            s=20,
            alpha=0.55,
        )

    ax_tbt.set_title(
        f"{data.variant.upper()} • {data.name}",
        fontsize=TITLE_FONTSIZE,
        pad=14,
    )
    configure_axis(ax_tbt, ylabel="TBT (s)")
    configure_axis(ax_ttft, xlabel="Time since start (s)", ylabel="TTFT (s)")

    ax_tbt.legend(
        title="Kind",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        loc="upper right",
    )
    ax_ttft.legend(
        title="Kind",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        loc="upper right",
    )


def add_latency_rolling_panel(fig: plt.Figure, outer_spec, data: ExperimentData) -> None:
    inner_spec: GridSpec = outer_spec.subgridspec(2, 1, hspace=0.08)
    ax_tbt = fig.add_subplot(inner_spec[0, 0])
    ax_ttft = fig.add_subplot(inner_spec[1, 0], sharex=ax_tbt)

    for kind, group in data.latency_df.groupby("kind"):
        ordered = group.sort_values("time_since_start_s")
        ax_tbt.plot(
            ordered["time_since_start_s"],
            ordered["avg_tbt"].rolling(ROLLING_WINDOW, min_periods=1).mean(),
            label=kind,
        )
        ax_ttft.plot(
            ordered["time_since_start_s"],
            ordered["ttft"].rolling(ROLLING_WINDOW, min_periods=1).mean(),
            label=kind,
        )

    ax_tbt.set_title(
        f"{data.variant.upper()} • {data.name}",
        fontsize=TITLE_FONTSIZE,
        pad=14,
    )
    configure_axis(ax_tbt, ylabel=f"TBT rolling mean (s)")
    configure_axis(ax_ttft, xlabel="Time since start (s)", ylabel=f"TTFT rolling mean (s)")

    ax_tbt.legend(
        title="Kind",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        loc="upper right",
    )
    ax_ttft.legend(
        title="Kind",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        loc="upper right",
    )


def add_throughput_panel(fig: plt.Figure, outer_spec, data: ExperimentData) -> None:
    if data.throughput_df.empty:
        ax = fig.add_subplot(outer_spec)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No throughput data",
            ha="center",
            va="center",
            fontsize=TITLE_FONTSIZE,
            color="red",
        )
        return

    inner_spec: GridSpec = outer_spec.subgridspec(3, 1, hspace=0.08)
    axes = [fig.add_subplot(inner_spec[i, 0]) for i in range(3)]

    metric_config = [
        ("prefill_throughput", "Prefill (t/s)"),
        ("decode_throughput", "Decode (t/s)"),
        ("total_throughput", "Total (t/s)"),
    ]

    for ax, (metric, label) in zip(axes, metric_config):
        for group_name, group_df in data.throughput_df.groupby("client_group"):
            ordered = group_df.sort_values("time_since_start_s")
            ax.plot(
                ordered["time_since_start_s"],
                ordered[metric].rolling(ROLLING_WINDOW, min_periods=1).mean(),
                label=group_name,
            )
        configure_axis(ax, ylabel=label)

    axes[0].set_title(
        f"{data.variant.upper()} • {data.name}",
        fontsize=TITLE_FONTSIZE,
        pad=14,
    )
    axes[-1].set_xlabel("Time since start (s)", fontsize=LABEL_FONTSIZE)
    axes[-1].tick_params(axis="x", labelsize=TICK_FONTSIZE)

    axes[0].legend(
        title="Client group",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        loc="upper right",
    )


def build_combined_figure(
    data_map: Dict[str, ExperimentData],
    *,
    title: str,
    tag: Optional[str],
    output_path: Path,
    panel_renderer,
) -> None:
    """Render a combined figure using the supplied panel renderer."""
    fig = plt.figure(figsize=(18, 14))
    outer = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.18)

    for idx, variant in enumerate(VARIANT_ORDER):
        spec = outer[idx // 2, idx % 2]
        if variant in data_map:
            panel_renderer(fig, spec, data_map[variant])
        else:
            ax = fig.add_subplot(spec)
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"No data for variant '{variant}'",
                ha="center",
                va="center",
                fontsize=TITLE_FONTSIZE,
                color="red",
            )

    suffix = f" ({tag})" if tag else ""
    fig.suptitle(f"{title}{suffix}", fontsize=SUPTITLE_FONTSIZE, y=0.98)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine experiment plots by regenerating them from CSV metrics. "
            "Provide one directory per traffic pattern (1s, 60s, vtc, plain)."
        )
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        type=Path,
        help="Paths to experiment directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to store combined figures.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional label appended to filenames and titles.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    variant_paths = collect_experiments(args.experiments)

    data_map: Dict[str, ExperimentData] = {}
    for variant, path in variant_paths.items():
        data_map[variant] = load_experiment(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    build_combined_figure(
        data_map,
        title="Latency per Token Over Time",
        tag=args.tag,
        output_path=args.output_dir
        / (f"latency_per_token_{args.tag}.png" if args.tag else "latency_per_token.png"),
        panel_renderer=add_latency_scatter_panel,
    )
    build_combined_figure(
        data_map,
        title="Rolling Average Latency per Token",
        tag=args.tag,
        output_path=args.output_dir
        / (f"rolling_latency_{args.tag}.png" if args.tag else "rolling_latency.png"),
        panel_renderer=add_latency_rolling_panel,
    )
    build_combined_figure(
        data_map,
        title="Rolling Average Throughput per Client Group",
        tag=args.tag,
        output_path=args.output_dir
        / (f"throughput_{args.tag}.png" if args.tag else "throughput.png"),
        panel_renderer=add_throughput_panel,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
