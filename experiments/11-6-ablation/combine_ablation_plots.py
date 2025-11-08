#!/usr/bin/env python3
"""
Generate combined plots for the 11-6 ablation studies.

The script scans the `results/ablation*` folders, loads the latest run for each
ablation parameter configuration, and produces:
  1. Latency figures with stacked TBT / TTFT panels per configuration.
  2. Throughput figures showing total tokens/sec per client group.
  3. Summary plots where the x-axis is the ablation parameter value and the
     y-axis is the median TBT.

Example:
    python combine_ablation_plots.py
    python combine_ablation_plots.py --ablations ablation1 ablation3 --rolling-window 7
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TITLE_FONTSIZE = 18
PANEL_TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12
LEGEND_TITLE_FONTSIZE = 13
SUPTITLE_FONTSIZE = 20


@dataclass
class LatencyData:
    df: pd.DataFrame
    t0: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp


@dataclass
class AblationVariant:
    name: str
    param_name: str
    param_value_label: str
    param_value_numeric: Optional[float]
    latency: pd.DataFrame
    throughput: pd.DataFrame


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine plots for all ablation studies in 11-6-ablation/results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Base directory that stores ablation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to store the generated figures (defaults to results-dir).",
    )
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=None,
        help="Optional subset of ablation folders to process (e.g. ablation1 ablation3).",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Window size (in samples) for rolling means.",
    )
    parser.add_argument(
        "--trim-start",
        type=float,
        default=30.0,
        help="Drop latency/throughput data before this many seconds since start (default 30).",
    )
    parser.add_argument(
        "--trim-end",
        type=float,
        default=None,
        help="Keep latency/throughput data up to this many seconds since start.",
    )
    parser.add_argument(
        "--good-client-max",
        type=int,
        default=20,
        help="User IDs up to this value are treated as good clients in throughput plots.",
    )
    parser.add_argument(
        "--total-client-max",
        type=int,
        default=600,
        help="Only users with IDs up to this value are considered for throughput aggregation.",
    )
    return parser.parse_args(list(argv))


def pick_latest_run_dir(variant_dir: Path) -> Optional[Path]:
    runs = sorted([p for p in variant_dir.iterdir() if p.is_dir()])
    return runs[-1] if runs else None


def pick_latest_csv(run_dir: Path, prefix: str) -> Optional[Path]:
    candidates = sorted(run_dir.glob(f"{prefix}_*.csv"))
    return candidates[-1] if candidates else None


def parse_param_metadata(dir_name: str) -> Tuple[str, str, Optional[float]]:
    """
    Extract the ablation parameter name and value from a directory like
    `abl2_rate_6`. Returns (param_name, label, numeric_value).
    """
    parts = dir_name.split("_", 2)
    if len(parts) < 3:
        return dir_name, dir_name, None
    _, param_name, raw_value = parts
    numeric_value: Optional[float] = None
    try:
        numeric_value = float(raw_value)
        if numeric_value.is_integer():
            numeric_value = int(numeric_value)
    except ValueError:
        numeric_value = None

    label = str(numeric_value) if numeric_value is not None else raw_value
    return param_name, label, numeric_value


def load_latency_data(
    run_dir: Path,
    *,
    trim_start: float,
    trim_end: Optional[float],
) -> LatencyData:
    lat_path = pick_latest_csv(run_dir, "latencies")
    if lat_path is None:
        raise FileNotFoundError(f"No latency CSV found in {run_dir}")

    df = pd.read_csv(lat_path)
    if "timestamp_start_iso" not in df.columns:
        raise ValueError(f"Latency CSV missing timestamp_start_iso column: {lat_path}")

    df["timestamp_start_iso"] = pd.to_datetime(df["timestamp_start_iso"], utc=True)
    df = df.sort_values("timestamp_start_iso")
    t0 = df["timestamp_start_iso"].iloc[0]
    df["time_since_start_s"] = (df["timestamp_start_iso"] - t0).dt.total_seconds()

    if trim_start > 0:
        df = df[df["time_since_start_s"] >= trim_start]
    if trim_end is not None:
        df = df[df["time_since_start_s"] <= trim_end]
    if df.empty:
        raise ValueError(
            f"No latency rows left after applying trims in {run_dir.name}:{lat_path.name}"
        )

    df["avg_tbt"] = np.where(
        df.get("completion_tokens", 0) > 0,
        df["duration_taft"] / df["completion_tokens"],
        np.nan,
    )
    df["ttft"] = df["duration_ttft"]
    if "kind" in df.columns:
        df["kind"] = df["kind"].astype(str)
    else:
        df["kind"] = "all"

    window_start = df["timestamp_start_iso"].min()
    window_end = df["timestamp_start_iso"].max()
    return LatencyData(df=df, t0=t0, window_start=window_start, window_end=window_end)


def compute_throughput_series(
    run_dir: Path,
    *,
    latency_window: LatencyData,
    trim_start: float,
    trim_end: Optional[float],
    good_client_max: int,
    total_client_max: int,
) -> pd.DataFrame:
    token_path = pick_latest_csv(run_dir, "token_usage")
    if token_path is None:
        return pd.DataFrame()

    token_df = pd.read_csv(token_path)
    if "timestamp_iso" not in token_df.columns:
        return pd.DataFrame()

    token_df["timestamp"] = pd.to_datetime(token_df["timestamp_iso"], utc=True)
    token_df = token_df.sort_values(["user_id", "timestamp"])
    token_df = token_df[token_df["user_id"] <= total_client_max].copy()
    if token_df.empty:
        return pd.DataFrame()

    token_df["elapsed_s"] = (
        token_df.groupby("user_id")["timestamp"].diff().dt.total_seconds()
    )
    token_df.loc[token_df["elapsed_s"] <= 0, "elapsed_s"] = np.nan

    totals = token_df["prefill_tokens"] + token_df["decode_tokens"]
    prefill_delta = token_df.groupby("user_id")["prefill_tokens"].diff().clip(lower=0)
    decode_delta = token_df.groupby("user_id")["decode_tokens"].diff().clip(lower=0)
    total_delta = (
        token_df.assign(total_tokens=totals)
        .groupby("user_id")["total_tokens"]
        .diff()
        .clip(lower=0)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        token_df["total_throughput"] = np.where(
            token_df["elapsed_s"] > 0,
            total_delta / token_df["elapsed_s"],
            np.nan,
        )

    token_df = token_df[
        (token_df["timestamp"] >= latency_window.window_start)
        & (token_df["timestamp"] <= latency_window.window_end)
    ].copy()
    if token_df.empty:
        return pd.DataFrame()

    token_df["time_since_start_s"] = (
        token_df["timestamp"] - latency_window.t0
    ).dt.total_seconds()
    if trim_start > 0:
        token_df = token_df[token_df["time_since_start_s"] >= trim_start]
    if trim_end is not None:
        token_df = token_df[token_df["time_since_start_s"] <= trim_end]
    if token_df.empty:
        return pd.DataFrame()

    max_user_id = min(int(token_df["user_id"].max()), total_client_max)
    bad_label = (
        f"Clients {good_client_max + 1}-{max_user_id}"
        if max_user_id > good_client_max
        else "Clients > good range"
    )
    token_df["client_group"] = np.where(
        token_df["user_id"] <= good_client_max,
        f"Clients 1-{good_client_max}",
        bad_label,
    )

    grouped = (
        token_df.dropna(subset=["total_throughput"])
        .groupby(["client_group", "timestamp"])["total_throughput"]
        .mean()
        .reset_index()
    )
    if grouped.empty:
        return pd.DataFrame()

    grouped["time_since_start_s"] = (
        grouped["timestamp"] - latency_window.t0
    ).dt.total_seconds()
    return grouped.sort_values(["client_group", "time_since_start_s"])


def collect_variants(
    group_dir: Path,
    *,
    trim_start: float,
    trim_end: Optional[float],
    good_client_max: int,
    total_client_max: int,
) -> List[AblationVariant]:
    variants: List[AblationVariant] = []
    for variant_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        run_dir = pick_latest_run_dir(variant_dir)
        if run_dir is None:
            print(f"[warn] No runs found in {variant_dir}")
            continue

        try:
            latency_window = load_latency_data(
                run_dir, trim_start=trim_start, trim_end=trim_end
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[warn] Skipping {variant_dir.name}: {exc}")
            continue

        throughput_df = compute_throughput_series(
            run_dir,
            latency_window=latency_window,
            trim_start=trim_start,
            trim_end=trim_end,
            good_client_max=good_client_max,
            total_client_max=total_client_max,
        )

        param_name, label, numeric_value = parse_param_metadata(variant_dir.name)
        variants.append(
            AblationVariant(
                name=variant_dir.name,
                param_name=param_name,
                param_value_label=label,
                param_value_numeric=numeric_value,
                latency=latency_window.df,
                throughput=throughput_df,
            )
        )
    return variants


def _layout(n_panels: int, max_cols: int = 2) -> Tuple[int, int]:
    ncols = min(max_cols, max(1, n_panels))
    nrows = math.ceil(n_panels / ncols)
    return nrows, ncols


def plot_latency_panels(
    group_name: str,
    variants: Sequence[AblationVariant],
    *,
    output_path: Path,
    rolling_window: int,
) -> None:
    nrows, ncols = _layout(len(variants))
    fig = plt.figure(figsize=(7 * ncols, 4.4 * nrows * 2))
    grid = fig.add_gridspec(nrows * 2, ncols, hspace=0.45, wspace=0.25)

    for idx, variant in enumerate(variants):
        base_row = (idx // ncols) * 2
        col = idx % ncols
        ax_tbt = fig.add_subplot(grid[base_row, col])
        ax_ttft = fig.add_subplot(grid[base_row + 1, col], sharex=ax_tbt)

        ordered = variant.latency.sort_values("time_since_start_s")
        tbt_series = ordered["avg_tbt"].rolling(rolling_window, min_periods=1).mean()
        ttft_series = ordered["ttft"].rolling(rolling_window, min_periods=1).mean()

        ax_tbt.plot(
            ordered["time_since_start_s"],
            tbt_series,
            linewidth=2.0,
        )
        ax_ttft.plot(
            ordered["time_since_start_s"],
            ttft_series,
            linewidth=2.0,
        )

        ax_tbt.set_ylabel("TBT (s)", fontsize=LABEL_FONTSIZE)
        ax_ttft.set_ylabel("TTFT (s)", fontsize=LABEL_FONTSIZE)
        ax_ttft.set_xlabel("Time since start (s)", fontsize=LABEL_FONTSIZE)

        for axis in (ax_tbt, ax_ttft):
            axis.set_title(
                f"{variant.param_name}={variant.param_value_label}",
                fontsize=PANEL_TITLE_FONTSIZE,
                pad=20,
            )
            axis.tick_params(axis="both", labelsize=TICK_FONTSIZE)
            axis.grid(True, alpha=0.3)

    fig.suptitle(
        f"{group_name}: TBT and TTFT",
        fontsize=SUPTITLE_FONTSIZE,
        y=0.995,
    )
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_throughput_panels(
    group_name: str,
    variants: Sequence[AblationVariant],
    *,
    output_path: Path,
    rolling_window: int,
) -> None:
    nrows, ncols = _layout(len(variants))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 4.4 * nrows),
        squeeze=False,
    )

    for idx, variant in enumerate(variants):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        if variant.throughput.empty:
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
            continue

        for group_name_client, group_df in variant.throughput.groupby("client_group"):
            ordered = group_df.sort_values("time_since_start_s")
            series = ordered["total_throughput"].rolling(
                rolling_window, min_periods=1
            ).mean()
            ax.plot(
                ordered["time_since_start_s"],
                series,
                linewidth=2.0,
            )

        ax.set_title(
            f"{variant.param_name}={variant.param_value_label}",
            fontsize=PANEL_TITLE_FONTSIZE,
            pad=18,
        )
        ax.set_xlabel("Time since start (s)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Tokens / s", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    total_axes = nrows * ncols
    for idx in range(len(variants), total_axes):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    fig.suptitle(
        f"{group_name}: Throughput",
        fontsize=SUPTITLE_FONTSIZE,
        y=0.995,
    )
    fig.subplots_adjust(hspace=0.5, wspace=0.25, top=0.92)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_tbt_vs_param(
    group_name: str,
    variants: Sequence[AblationVariant],
    *,
    output_path: Path,
    rolling_window: int,
) -> None:
    summaries: List[Dict[str, float]] = []
    labels: List[str] = []
    numeric_positions: List[Tuple[float, str]] = []

    for variant in variants:
        df = variant.latency.dropna(subset=["avg_tbt"])
        summary = {"Overall": df["avg_tbt"].median()}
        for kind, group in df.groupby("kind"):
            summary[kind] = group["avg_tbt"].median()
        summaries.append(summary)
        labels.append(variant.param_value_label)

        position = (
            float(variant.param_value_numeric)
            if variant.param_value_numeric is not None
            else len(numeric_positions)
        )
        numeric_positions.append((position, variant.param_value_label))

    unique_keys: List[str] = []
    for summary in summaries:
        for key in summary:
            if key not in unique_keys:
                unique_keys.append(key)

    x_vals = [pos for pos, _ in numeric_positions]
    sort_indices = sorted(range(len(x_vals)), key=lambda idx: x_vals[idx])
    sorted_positions = [numeric_positions[idx] for idx in sort_indices]
    sorted_summaries = [summaries[idx] for idx in sort_indices]

    fig, ax = plt.subplots(figsize=(max(6, 4 + 1.2 * len(variants)), 5))

    for key in unique_keys:
        xs = []
        ys = []
        for (x_pos, _), summary in zip(sorted_positions, sorted_summaries):
            value = summary.get(key)
            if value is not None and not np.isnan(value):
                xs.append(x_pos)
                ys.append(value)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.0)

    ax.set_xticks([pos for pos, _ in sorted_positions])
    ax.set_xticklabels([label for _, label in sorted_positions], fontsize=TICK_FONTSIZE)
    param_name = variants[0].param_name if variants else "parameter"
    ax.set_xlabel(f"{param_name}", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Median TBT (s)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{group_name}: TBT vs {param_name}", fontsize=SUPTITLE_FONTSIZE)

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir
    output_root = args.output_dir or results_dir

    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    available_groups = sorted(
        [p.name for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("ablation")]
    )
    target_groups = args.ablations or available_groups

    if not target_groups:
        raise SystemExit("No ablation directories found to process.")

    for group_name in target_groups:
        group_dir = results_dir / group_name
        if not group_dir.exists():
            print(f"[warn] Skipping missing group {group_name}")
            continue

        variants = collect_variants(
            group_dir,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
            good_client_max=args.good_client_max,
            total_client_max=args.total_client_max,
        )
        if not variants:
            print(f"[warn] No valid runs found under {group_name}")
            continue

        variants.sort(
            key=lambda var: (
                0 if var.param_value_numeric is not None else 1,
                var.param_value_numeric if var.param_value_numeric is not None else var.name,
            )
        )

        group_output = output_root / group_name
        group_output.mkdir(parents=True, exist_ok=True)

        plot_latency_panels(
            group_name,
            variants,
            output_path=group_output / f"{group_name}_latency.png",
            rolling_window=args.rolling_window,
        )
        plot_throughput_panels(
            group_name,
            variants,
            output_path=group_output / f"{group_name}_throughput.png",
            rolling_window=args.rolling_window,
        )
        plot_tbt_vs_param(
            group_name,
            variants,
            output_path=group_output / f"{group_name}_tbt_vs_param.png",
            rolling_window=args.rolling_window,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
