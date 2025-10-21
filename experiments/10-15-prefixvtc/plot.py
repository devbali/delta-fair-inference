import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Settings ---
CSV_PATH = "/home/devbali/fairinf/delta-fair-inference/experiments/10-15-prefixvtc/latencies_20251021_203911.csv"
GPU_CSV_PATH = CSV_PATH.replace("latencies_", "gpu_metrics_")
PROMETHEUS_CSV_PATH = CSV_PATH.replace("latencies_", "prometheus_metrics_")
TOKEN_USAGE_CSV_PATH = CSV_PATH.replace("latencies_", "token_usage_")
SGLANG_LOG_PATH = "sglang_log_decode.csv"
METRICS_SOURCE = "auto"  # Options: "sglang", "prometheus", "auto"
SCATTER_OUT = "latency_per_token_over_time.png"
BARS_OUT = "token_averages_by_kind.png"
ROLLING_OUT = "rolling_avg_latency_per_token.png"
THROUGHPUT_ROLLING_OUT = "rolling_avg_throughput_per_client_group.png"
METRICS_COMBINED_OUT = "latency_and_metrics.png"

METRIC_CONFIG = [
    ("token_usage", "Token Usage"),
    ("throughput", "Generation Throughput\n(tokens/sec)"),
    ("running_reqs", "Number of\nRunning Requests"),
    ("queue_reqs", "Number of\nQueued Requests"),
]
PROMETHEUS_METRIC_PREFIXES = {
    "token_usage": "sglang:token_usage{engine_type=",
    "throughput": "sglang:gen_throughput{engine_type=",
    "running_reqs": "sglang:num_running_reqs{engine_type=",
    "queue_reqs": "sglang:num_queue_reqs{engine_type=",
}
ROLLING_WINDOW = 5  # rolling mean window in samples for latency
METRICS_WINDOW = '5s'  # rolling window for metrics
CUT_FIRST_SECONDS = 90  # cut first 90 seconds from latency data
PLOT_TIME_START_S = 250  # Optional lower bound (seconds since first timestamp)
PLOT_TIME_END_S = None    # Optional upper bound (seconds since first timestamp)


def filter_time_window(df, time_col):
    """Filter dataframe to the configured plotting time window."""
    if df.empty:
        return df
    filtered = df
    if PLOT_TIME_START_S is not None:
        filtered = filtered[filtered[time_col] >= PLOT_TIME_START_S]
    if PLOT_TIME_END_S is not None:
        filtered = filtered[filtered[time_col] <= PLOT_TIME_END_S]
    return filtered


def load_sglang_metrics(latency_start, latency_end, original_t0):
    """Load metrics from the sglang decode CSV if present."""
    if not os.path.exists(SGLANG_LOG_PATH):
        return {}

    raw_df = pd.read_csv(SGLANG_LOG_PATH)
    if raw_df.empty:
        return {}

    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], unit="s", utc=True)
    raw_df = raw_df.sort_values("timestamp")
    raw_df = raw_df[
        (raw_df["timestamp"] >= latency_start) &
        (raw_df["timestamp"] <= latency_end)
    ].copy()
    if raw_df.empty:
        return {}

    raw_df["time_since_start_s"] = (raw_df["timestamp"] - original_t0).dt.total_seconds()
    raw_df = filter_time_window(raw_df, "time_since_start_s")
    if raw_df.empty:
        return {}

    metrics_data = {}
    for metric_name, _ in METRIC_CONFIG:
        if metric_name not in raw_df.columns:
            continue
        metric_df = raw_df[["timestamp", "time_since_start_s", metric_name]].rename(
            columns={metric_name: "value"}
        )
        metric_df = metric_df.set_index("timestamp")
        metrics_data[metric_name] = metric_df

    return metrics_data


def load_prometheus_metrics(latency_start, latency_end, original_t0):
    """Load metrics from the Prometheus scrape CSV if present."""
    if not os.path.exists(PROMETHEUS_CSV_PATH):
        return {}

    prometheus_df = pd.read_csv(PROMETHEUS_CSV_PATH)
    if prometheus_df.empty:
        return {}

    prometheus_df["timestamp"] = pd.to_datetime(prometheus_df["timestamp"], utc=True)
    prometheus_df = prometheus_df.sort_values("timestamp")

    metrics_data = {}
    for metric_name, metric_prefix in PROMETHEUS_METRIC_PREFIXES.items():
        metric_rows = prometheus_df[
            prometheus_df["metric_name"].str.startswith(metric_prefix)
        ].copy()
        if metric_rows.empty:
            continue

        metric_rows["value"] = pd.to_numeric(metric_rows["value"], errors="coerce")
        metric_rows = metric_rows.dropna(subset=["value"])
        metric_rows = metric_rows[
            (metric_rows["timestamp"] >= latency_start) &
            (metric_rows["timestamp"] <= latency_end)
        ].copy()
        if metric_rows.empty:
            continue

        metric_rows["time_since_start_s"] = (
            metric_rows["timestamp"] - original_t0
        ).dt.total_seconds()
        metric_rows = filter_time_window(metric_rows, "time_since_start_s")
        if metric_rows.empty:
            continue

        metric_rows = metric_rows.set_index("timestamp")[["value", "time_since_start_s"]]
        metrics_data[metric_name] = metric_rows

    return metrics_data


def resolve_metrics_data(selection, latency_start, latency_end, original_t0):
    """Return the metrics data dictionary and the source used."""
    normalized = selection.lower()
    if normalized not in {"auto", "sglang", "prometheus"}:
        raise ValueError(f"Unexpected METRICS_SOURCE '{selection}'.")

    if normalized == "auto":
        sources_to_try = ("sglang", "prometheus")
    else:
        sources_to_try = (normalized,)

    for source in sources_to_try:
        if source == "sglang":
            metrics_data = load_sglang_metrics(latency_start, latency_end, original_t0)
        else:
            metrics_data = load_prometheus_metrics(latency_start, latency_end, original_t0)

        if metrics_data:
            return source, metrics_data

    return None, {}

# --- Load & prep ---
# Load and prepare latency data
df = pd.read_csv(CSV_PATH)
df["timestamp_start_iso"] = pd.to_datetime(df["timestamp_start_iso"], utc=True)

# Print timestamp for debugging
print("Sample latency timestamp:", df["timestamp_start_iso"].iloc[0])

# Calculate original time since start for latency data
original_t0 = df["timestamp_start_iso"].iloc[0]
df["time_since_start_s"] = (df["timestamp_start_iso"] - original_t0).dt.total_seconds()

if PLOT_TIME_START_S is not None or PLOT_TIME_END_S is not None:
    print(
        "Applying plotting time window "
        f"[{PLOT_TIME_START_S if PLOT_TIME_START_S is not None else 'start'}, "
        f"{PLOT_TIME_END_S if PLOT_TIME_END_S is not None else 'end'}] seconds."
    )

# Cut first 90 seconds from latency data but preserve original timing
df = df[df["time_since_start_s"] >= CUT_FIRST_SECONDS].copy()
df = filter_time_window(df, "time_since_start_s")
if df.empty:
    raise ValueError(
        "No latency data left after applying CUT_FIRST_SECONDS and plotting time window. "
        "Adjust PLOT_TIME_START_S and PLOT_TIME_END_S."
    )
print(f"\nLatency data after {CUT_FIRST_SECONDS}s cut:")
print(f"Start: {df['timestamp_start_iso'].min()}")
print(f"End: {df['timestamp_start_iso'].max()}")

latency_start = df["timestamp_start_iso"].min()
latency_end = df["timestamp_start_iso"].max()

# Load GPU metrics
gpu_df = pd.read_csv(GPU_CSV_PATH)
gpu_df["timestamp"] = pd.to_datetime(gpu_df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", utc=True)
gpu_df = gpu_df[
    (gpu_df["timestamp"] >= latency_start) & 
    (gpu_df["timestamp"] <= latency_end)
]
gpu_df["time_since_start_s"] = (gpu_df["timestamp"] - original_t0).dt.total_seconds()
gpu_df = filter_time_window(gpu_df, "time_since_start_s")

# Load metrics according to configured source
metrics_source_used, metrics_data = resolve_metrics_data(
    METRICS_SOURCE, latency_start, latency_end, original_t0
)

throughput_grouped = pd.DataFrame()
if os.path.exists(TOKEN_USAGE_CSV_PATH):
    # Load token usage metrics and compute per-client throughputs
    token_usage_df = pd.read_csv(TOKEN_USAGE_CSV_PATH)
    token_usage_df["timestamp"] = pd.to_datetime(token_usage_df["timestamp_iso"], utc=True)
    token_usage_df = token_usage_df.sort_values(["user_id", "timestamp"])

    # Keep only the clients we care about (1-20)
    token_usage_df = token_usage_df[token_usage_df["user_id"].between(1, 20)].copy()

    # Compute elapsed seconds between samples per client
    token_usage_df["elapsed_s"] = (
        token_usage_df.groupby("user_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )
    token_usage_df.loc[token_usage_df["elapsed_s"] <= 0, "elapsed_s"] = np.nan

    # Compute token deltas per client
    prefill_delta = (
        token_usage_df.groupby("user_id")["prefill_tokens"]
        .diff()
        .clip(lower=0)
    )
    decode_delta = (
        token_usage_df.groupby("user_id")["decode_tokens"]
        .diff()
        .clip(lower=0)
    )
    total_tokens = token_usage_df["prefill_tokens"] + token_usage_df["decode_tokens"]
    total_delta = (
        token_usage_df.assign(total_tokens=total_tokens)
        .groupby("user_id")["total_tokens"]
        .diff()
        .clip(lower=0)
    )

    # Convert deltas to throughput (tokens per second)
    token_usage_df["prefill_throughput"] = np.where(
        token_usage_df["elapsed_s"] > 0,
        prefill_delta / token_usage_df["elapsed_s"],
        np.nan
    )
    token_usage_df["decode_throughput"] = np.where(
        token_usage_df["elapsed_s"] > 0,
        decode_delta / token_usage_df["elapsed_s"],
        np.nan
    )
    token_usage_df["total_throughput"] = np.where(
        token_usage_df["elapsed_s"] > 0,
        total_delta / token_usage_df["elapsed_s"],
        np.nan
    )

    token_usage_df["time_since_start_s"] = (token_usage_df["timestamp"] - original_t0).dt.total_seconds()
    token_usage_df = token_usage_df[
        (token_usage_df["timestamp"] >= latency_start) &
        (token_usage_df["timestamp"] <= latency_end)
    ].copy()
    token_usage_df = filter_time_window(token_usage_df, "time_since_start_s")

    token_usage_df["client_group"] = np.where(
        token_usage_df["user_id"] <= 15,
        "Good clients (1-15)",
        "Bad clients (16-20)"
    )

    # Aggregate throughput by timestamp per client group
    throughput_metrics = ["prefill_throughput", "decode_throughput", "total_throughput"]
    throughput_grouped = (
        token_usage_df.dropna(subset=throughput_metrics)
        .groupby(["client_group", "timestamp"])
        [throughput_metrics]
        .mean()
        .reset_index()
    )
    throughput_grouped["time_since_start_s"] = (
        throughput_grouped["timestamp"] - original_t0
    ).dt.total_seconds()
    throughput_grouped = filter_time_window(throughput_grouped, "time_since_start_s")
else:
    print(f"Token usage CSV not found at {TOKEN_USAGE_CSV_PATH}; throughput plots skipped.")

print("\nAfter synchronization:")
print(f"Reference t0 (experiment start): {original_t0}")
print(f"GPU data points: {len(gpu_df)}")
if metrics_source_used:
    print(f"Metrics source: {metrics_source_used}")
    for metric_name in sorted(metrics_data):
        print(f"{metric_name} data points: {len(metrics_data[metric_name])}")
else:
    print("Metrics source: none (no metrics data available)")

print("\nTime ranges relative to experiment start:")
print(f"Latency time range (after {CUT_FIRST_SECONDS}s cut): {df['time_since_start_s'].min():.1f}s to {df['time_since_start_s'].max():.1f}s")
print(f"GPU metrics range: {gpu_df['time_since_start_s'].min():.1f}s to {gpu_df['time_since_start_s'].max():.1f}s")
if metrics_source_used:
    for metric_name in sorted(metrics_data):
        metric_df = metrics_data[metric_name]
        print(
            f"{metric_name} metrics range: "
            f"{metric_df['time_since_start_s'].min():.1f}s to "
            f"{metric_df['time_since_start_s'].max():.1f}s"
        )

# Calculate Time Between Tokens (TBT) from TAFT
df["avg_tbt"] = np.where(
    df["completion_tokens"] > 0,
    df["duration_taft"] / df["completion_tokens"],
    np.nan
)
df["kind"] = df["kind"].astype(str)
df["ttft"] = df["duration_ttft"]  # Store TTFT for plotting

# ======================================================
# 1) Scatter plots: TBT and TTFT vs. time (filtered)
# ======================================================
# Create a figure with two subplots sharing x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Latency Metrics Over Time Since Start", y=0.95)

# Plot TBT
for kind, group in df.groupby("kind"):
    ax1.scatter(
        group["time_since_start_s"],
        group["avg_tbt"],
        label=kind,
        s=20,
        alpha=0.5
    )
ax1.set_ylabel("Time Between Tokens (s)")
ax1.legend(title="Kind")
ax1.grid(True, alpha=0.3)

# Plot TTFT
for kind, group in df.groupby("kind"):
    ax2.scatter(
        group["time_since_start_s"],
        group["ttft"],
        label=kind,
        s=20,
        alpha=0.5
    )
ax2.set_xlabel("Time since start (seconds)")
ax2.set_ylabel("Time To First Token (s)")
ax2.legend(title="Kind")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SCATTER_OUT, dpi=300, bbox_inches='tight')
plt.close()


# ======================================================
# 2) Grouped bars: averages with error bars (std deviation)
# ======================================================
metrics = ["prompt_tokens", "completion_tokens"]  # Remove total_tokens since it's not in the data
agg = df.groupby("kind")[metrics].agg(["mean", "std"])
kinds = list(agg.index)
x = np.arange(len(metrics))
width = 0.8 / max(len(kinds), 1)

plt.figure(figsize=(10, 6))
for i, kind in enumerate(kinds):
    means = [agg.loc[kind, (m, "mean")] for m in metrics]
    stds  = [agg.loc[kind, (m, "std")]  for m in metrics]
    offsets = x + (i - (len(kinds)-1)/2.0) * width
    plt.bar(offsets, means, width=width, label=kind, yerr=stds, capsize=4)

plt.xticks(x, ["Prompt tokens", "Completion tokens"])
plt.ylabel("Average tokens")
plt.title("Average Token Counts by Kind (error bars = ±1 SD)")
plt.legend(title="Kind")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(BARS_OUT, dpi=300)
plt.close()

# ======================================================
# 3) Rolling average (TBT and TTFT) by kind (filtered)
# ======================================================
# Create stacked plot with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"Rolling Average of Latency Metrics by Kind (≥{CUT_FIRST_SECONDS}s)", y=0.95)

# Plot TBT
for kind, group in df.groupby("kind"):
    group_sorted = group.sort_values("time_since_start_s")
    rolling_tbt = group_sorted["avg_tbt"].rolling(ROLLING_WINDOW).mean()
    ax1.plot(group_sorted["time_since_start_s"], rolling_tbt, label=kind)
ax1.set_ylabel(f"Rolling mean TBT (s, window={ROLLING_WINDOW})")
ax1.legend(title="Kind")
ax1.grid(True, alpha=0.3)

# Plot TTFT
for kind, group in df.groupby("kind"):
    group_sorted = group.sort_values("time_since_start_s")
    rolling_ttft = group_sorted["ttft"].rolling(ROLLING_WINDOW).mean()
    ax2.plot(group_sorted["time_since_start_s"], rolling_ttft, label=kind)
ax2.set_xlabel("Time since start (seconds)")
ax2.set_ylabel(f"Rolling mean TTFT (s, window={ROLLING_WINDOW})")
ax2.legend(title="Kind")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ROLLING_OUT, dpi=300, bbox_inches='tight')
plt.close()

# ======================================================
# 4) Rolling average throughput by client group
# ======================================================
if not throughput_grouped.empty:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Rolling Average Throughput by Client Group (≥{CUT_FIRST_SECONDS}s)",
        y=0.95
    )

    metric_config = [
        ("prefill_throughput", "Prefill throughput (tokens/sec)"),
        ("decode_throughput", "Decode throughput (tokens/sec)"),
        ("total_throughput", "Total throughput (tokens/sec)")
    ]

    for ax, (metric, label) in zip(axes, metric_config):
        for group_name, group_df in throughput_grouped.groupby("client_group"):
            group_sorted = group_df.sort_values("time_since_start_s")
            rolling_series = group_sorted[metric].rolling(ROLLING_WINDOW, min_periods=1).mean()
            ax.plot(group_sorted["time_since_start_s"], rolling_series, label=group_name)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[0].legend(title="Client group")
    axes[-1].set_xlabel("Time since start (seconds)")

    plt.tight_layout()
    plt.savefig(THROUGHPUT_ROLLING_OUT, dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("No throughput data available for the selected range.")

# Metrics already time-aligned above
if metrics_source_used:
    print("\nNumber of metrics data points in the latency period:")
    for metric_name in sorted(metrics_data):
        print(f"  {metric_name}: {len(metrics_data[metric_name])}")
else:
    print("\nNo metrics data points available in the latency period.")

# Create metrics plot with 7 subplots (TBT, TTFT, and other metrics)
fig, axes = plt.subplots(7, 1, figsize=(12, 24), sharex=True)
lat_tbt_ax, lat_ttft_ax = axes[0], axes[1]
metric_axes = axes[2:6]
gpu_ax = axes[6]

# 1. TBT plot
for kind, group in df.groupby("kind"):
    group_sorted = group.sort_values("time_since_start_s")
    rolling_tbt = group_sorted["avg_tbt"].rolling(ROLLING_WINDOW).mean()
    lat_tbt_ax.plot(group_sorted["time_since_start_s"], rolling_tbt, label=kind)
lat_tbt_ax.set_ylabel("TBT (seconds)")
lat_tbt_ax.legend(title="Kind")
lat_tbt_ax.grid(True, alpha=0.3)

# 2. TTFT plot
for kind, group in df.groupby("kind"):
    group_sorted = group.sort_values("time_since_start_s")
    rolling_ttft = group_sorted["ttft"].rolling(ROLLING_WINDOW).mean()
    lat_ttft_ax.plot(group_sorted["time_since_start_s"], rolling_ttft, label=kind)
lat_ttft_ax.set_ylabel("TTFT (seconds)")
lat_ttft_ax.legend(title="Kind")
lat_ttft_ax.grid(True, alpha=0.3)

# 3-6. Metrics plots (token usage, throughput, running/queued requests)
for ax, (metric_name, ylabel) in zip(metric_axes, METRIC_CONFIG):
    metric_df = metrics_data.get(metric_name)
    ax.set_ylabel(ylabel)
    if metric_df is None or metric_df.empty:
        ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, alpha=0.3)
        continue

    rolling_series = metric_df["value"].rolling(window=METRICS_WINDOW, min_periods=1).mean()
    ax.plot(
        metric_df["time_since_start_s"].to_numpy(),
        rolling_series.to_numpy(),
        label=f"{metric_name.replace('_', ' ').title()} ({METRICS_WINDOW} avg)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

# 7. GPU Memory plot
gpu_ax.plot(gpu_df["time_since_start_s"], gpu_df["mem_used_MiB"], label='GPU Memory', alpha=0.6)
gpu_ax.set_ylabel("GPU Memory (MiB)")
gpu_ax.set_xlabel("Time since start (seconds)")
gpu_ax.legend()
gpu_ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(METRICS_COMBINED_OUT, dpi=300, bbox_inches='tight')
plt.close()

print("\nData ranges after synchronization:")
print(f"Latency time range: {df['time_since_start_s'].min():.1f}s to {df['time_since_start_s'].max():.1f}s")
if metrics_source_used:
    for metric_name in sorted(metrics_data):
        metric_df = metrics_data[metric_name]
        print(
            f"{metric_name} time range: "
            f"{metric_df['time_since_start_s'].min():.1f}s to "
            f"{metric_df['time_since_start_s'].max():.1f}s"
        )
else:
    print("Metrics time range: n/a (no metrics data)")
