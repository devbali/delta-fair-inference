import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Settings ---
CSV_PATH = "/home/devbali/fairinf/delta-fair-inference/experiments/10-27-prefixvtc-plain-300/latencies_20251028_044209.csv"
GPU_CSV_PATH = CSV_PATH.replace("latencies_", "gpu_metrics_")
TOKEN_USAGE_CSV_PATH = CSV_PATH.replace("latencies_", "token_usage_")
SGLANG_LOG_PATH = "sglang_log_decode.csv"
SCATTER_OUT = "latency_per_token_over_time.png"
BARS_OUT = "token_averages_by_kind.png"
ROLLING_OUT = "rolling_avg_latency_per_token.png"
THROUGHPUT_ROLLING_OUT = "rolling_avg_throughput_per_client_group.png"
METRICS_COMBINED_OUT = "latency_and_metrics.png"

METRIC_LABELS = ["Token Usage", "Generation Throughput", "Running Requests"]
ROLLING_WINDOW = 5  # rolling mean window in samples for latency
METRICS_WINDOW = '5s'  # rolling window for metrics
CUT_FIRST_SECONDS = 90  # cut first 90 seconds from latency data
PLOT_TIME_START_S = None  # Optional lower bound (seconds since first timestamp)
PLOT_TIME_END_S = 300    # Optional upper bound (seconds since first timestamp)


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

# Load GPU metrics
gpu_df = pd.read_csv(GPU_CSV_PATH)
gpu_df["timestamp"] = pd.to_datetime(gpu_df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", utc=True)

# Load sglang metrics
sglang_df = pd.read_csv(SGLANG_LOG_PATH)
sglang_df["timestamp"] = pd.to_datetime(sglang_df["timestamp"], unit='s', utc=True)

# Filter other metrics to match latency timestamp range
latency_start = df["timestamp_start_iso"].min()
latency_end = df["timestamp_start_iso"].max()

gpu_df = gpu_df[
    (gpu_df["timestamp"] >= latency_start) & 
    (gpu_df["timestamp"] <= latency_end)
]

sglang_df = sglang_df[
    (sglang_df["timestamp"] >= latency_start) & 
    (sglang_df["timestamp"] <= latency_end)
]

# Calculate time_since_start_s for all dataframes using original t0
gpu_df["time_since_start_s"] = (gpu_df["timestamp"] - original_t0).dt.total_seconds()
sglang_df["time_since_start_s"] = (sglang_df["timestamp"] - original_t0).dt.total_seconds()
gpu_df = filter_time_window(gpu_df, "time_since_start_s")
sglang_df = filter_time_window(sglang_df, "time_since_start_s")

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
        token_usage_df["user_id"] <= 20,
        "Good clients (1-20)",
        "Bad clients (NA)"
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
print(f"Sglang data points: {len(sglang_df)}")

print("\nTime ranges relative to experiment start:")
print(f"Latency time range (after {CUT_FIRST_SECONDS}s cut): {df['time_since_start_s'].min():.1f}s to {df['time_since_start_s'].max():.1f}s")
print(f"GPU metrics range: {gpu_df['time_since_start_s'].min():.1f}s to {gpu_df['time_since_start_s'].max():.1f}s")
print(f"Sglang metrics range: {sglang_df['time_since_start_s'].min():.1f}s to {sglang_df['time_since_start_s'].max():.1f}s")

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

# Sglang metrics already synchronized above

# Already converted above

print(f"\nNumber of metrics data points in the latency period: {len(sglang_df)}")

# Create metrics plot with 7 subplots (TBT, TTFT, and other metrics)
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(12, 24), sharex=True)

# 1. TBT plot
for kind, group in df.groupby("kind"):
    group_sorted = group.sort_values("time_since_start_s")
    rolling_tbt = group_sorted["avg_tbt"].rolling(ROLLING_WINDOW).mean()
    ax1.plot(group_sorted["time_since_start_s"], rolling_tbt, label=kind)
ax1.set_ylabel("TBT (seconds)")
ax1.legend(title="Kind")
ax1.grid(True, alpha=0.3)

# 2. TTFT plot
for kind, group in df.groupby("kind"):
    group_sorted = group.sort_values("time_since_start_s")
    rolling_ttft = group_sorted["ttft"].rolling(ROLLING_WINDOW).mean()
    ax2.plot(group_sorted["time_since_start_s"], rolling_ttft, label=kind)
ax2.set_ylabel("TTFT (seconds)")
ax2.legend(title="Kind")
ax2.grid(True, alpha=0.3)

# Sort sglang metrics by time for rolling average
sglang_df = sglang_df.sort_values("timestamp")

# Set timestamp as index for time-based rolling
sglang_df = sglang_df.set_index('timestamp')

# Use metrics from the same start time as latency
metrics_display_df = sglang_df  # Already filtered by CUT_FIRST_SECONDS

# 3. Token Usage plot
rolling_usage = sglang_df["token_usage"].rolling(window=METRICS_WINDOW, min_periods=1).mean()
ax3.plot(metrics_display_df["time_since_start_s"], rolling_usage[metrics_display_df.index], label=f"Token Usage ({METRICS_WINDOW} avg)")
ax3.set_ylabel("Token Usage")
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Generation Throughput plot
rolling_throughput = sglang_df["throughput"].rolling(window=METRICS_WINDOW, min_periods=1).mean()
ax4.plot(metrics_display_df["time_since_start_s"], rolling_throughput[metrics_display_df.index], label=f"Generation Throughput ({METRICS_WINDOW} avg)")
ax4.set_ylabel("Generation Throughput\n(tokens/sec)")
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Running Requests plot
rolling_reqs = sglang_df["running_reqs"].rolling(window=METRICS_WINDOW, min_periods=1).mean()
ax5.plot(metrics_display_df["time_since_start_s"], rolling_reqs[metrics_display_df.index], label=f"Running Requests ({METRICS_WINDOW} avg)")
ax5.set_ylabel("Number of\nRunning Requests")
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Queued Requests plot
rolling_queue = sglang_df["queue_reqs"].rolling(window=METRICS_WINDOW, min_periods=1).mean()
ax6.plot(metrics_display_df["time_since_start_s"], rolling_queue[metrics_display_df.index], label=f"Queued Requests ({METRICS_WINDOW} avg)")
ax6.set_ylabel("Number of\nQueued Requests")
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. GPU Memory plot
ax7.plot(gpu_df["time_since_start_s"], gpu_df["mem_used_MiB"], label='GPU Memory', alpha=0.6)
ax7.set_ylabel("GPU Memory (MiB)")
ax7.set_xlabel("Time since start (seconds)")
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(METRICS_COMBINED_OUT, dpi=300, bbox_inches='tight')
plt.close()

print("\nData ranges after synchronization:")
print(f"Latency time range: {df['time_since_start_s'].min():.1f}s to {df['time_since_start_s'].max():.1f}s")
print(f"Metrics time range: {sglang_df['time_since_start_s'].min():.1f}s to {sglang_df['time_since_start_s'].max():.1f}s")
