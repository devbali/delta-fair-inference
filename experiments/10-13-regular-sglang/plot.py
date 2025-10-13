import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Settings ---
CSV_PATH = "/home/devbali/fairinf/10-13-regular-sglang/latencies_20251013_221137.csv"
GPU_CSV_PATH = CSV_PATH.replace("latencies_", "gpu_metrics_")
PROMETHEUS_CSV_PATH = CSV_PATH.replace("latencies_", "prometheus_metrics_")
SCATTER_OUT = "latency_per_token_over_time.png"
BARS_OUT = "token_averages_by_kind.png"
ROLLING_OUT = "rolling_avg_latency_per_token.png"
METRICS_COMBINED_OUT = "latency_and_metrics.png"

ROLLING_WINDOW = 5  # rolling mean window in samples for latency
METRICS_WINDOW = '5s'  # rolling window for metrics
CUT_FIRST_SECONDS = 90  # cut first 90 seconds from latency data
SCATTER_OUT = "latency_per_token_over_time.png"
BARS_OUT = "token_averages_by_kind.png"
ROLLING_OUT = "rolling_avg_latency_per_token.png"
METRICS_COMBINED_OUT = "latency_and_metrics.png"

METRIC_LABELS = ["Token Usage", "Generation Throughput", "Running Requests"]
ROLLING_WINDOW = 5  # rolling mean window in samples for latency
METRICS_WINDOW = '5s'  # rolling window for metrics
CUT_FIRST_SECONDS = 90  # cut first 90 seconds from latency data

# --- Load & prep ---
# Load and prepare latency data
df = pd.read_csv(CSV_PATH)
df["timestamp_start_iso"] = pd.to_datetime(df["timestamp_start_iso"], utc=True)

# Print timestamp for debugging
print("Sample latency timestamp:", df["timestamp_start_iso"].iloc[0])

# Calculate original time since start for latency data
original_t0 = df["timestamp_start_iso"].iloc[0]
df["time_since_start_s"] = (df["timestamp_start_iso"] - original_t0).dt.total_seconds()

# Cut first 90 seconds from latency data but preserve original timing
df = df[df["time_since_start_s"] >= CUT_FIRST_SECONDS].copy()
print(f"\nLatency data after {CUT_FIRST_SECONDS}s cut:")
print(f"Start: {df['timestamp_start_iso'].min()}")
print(f"End: {df['timestamp_start_iso'].max()}")

# Load GPU metrics
gpu_df = pd.read_csv(GPU_CSV_PATH)
gpu_df["timestamp"] = pd.to_datetime(gpu_df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", utc=True)

# Load Prometheus metrics
prometheus_df = pd.read_csv(PROMETHEUS_CSV_PATH)
prometheus_df["timestamp"] = pd.to_datetime(prometheus_df["timestamp"], utc=True)

# Process Prometheus metrics to get the time series data
# Create separate dataframes for each metric type
metrics_map = {
    "token_usage": "sglang:token_usage{engine_type=",
    "throughput": "sglang:gen_throughput{engine_type=",
    "running_reqs": "sglang:num_running_reqs{engine_type=",
    "queue_reqs": "sglang:num_queue_reqs{engine_type="
}

metrics_data = {}
for metric_name, metric_prefix in metrics_map.items():
    metric_rows = prometheus_df[prometheus_df["metric_name"].str.startswith(metric_prefix)]
    if not metric_rows.empty:
        metrics_data[metric_name] = pd.DataFrame({
            'timestamp': metric_rows['timestamp'],
            'value': metric_rows['value'].astype(float)
        }).set_index('timestamp')

# Filter other metrics to match latency timestamp range
latency_start = df["timestamp_start_iso"].min()
latency_end = df["timestamp_start_iso"].max()

gpu_df = gpu_df[
    (gpu_df["timestamp"] >= latency_start) & 
    (gpu_df["timestamp"] <= latency_end)
]

for metric_name in metrics_data:
    metrics_data[metric_name] = metrics_data[metric_name][
        (metrics_data[metric_name].index >= latency_start) & 
        (metrics_data[metric_name].index <= latency_end)
    ]

# Calculate time_since_start_s for all dataframes using original t0
gpu_df["time_since_start_s"] = (gpu_df["timestamp"] - original_t0).dt.total_seconds()
for metric_name in metrics_data:
    metrics_data[metric_name]['time_since_start_s'] = pd.Series(
        index=metrics_data[metric_name].index,
        data=(metrics_data[metric_name].index - original_t0).total_seconds()
    )

print("\nAfter synchronization:")
print(f"Reference t0 (experiment start): {original_t0}")
print(f"GPU data points: {len(gpu_df)}")
for metric_name in metrics_data:
    print(f"{metric_name} data points: {len(metrics_data[metric_name])}")

print("\nTime ranges relative to experiment start:")
print(f"Latency time range (after {CUT_FIRST_SECONDS}s cut): {df['time_since_start_s'].min():.1f}s to {df['time_since_start_s'].max():.1f}s")
print(f"GPU metrics range: {gpu_df['time_since_start_s'].min():.1f}s to {gpu_df['time_since_start_s'].max():.1f}s")
for metric_name in metrics_data:
    data = metrics_data[metric_name]
    print(f"{metric_name} range: {data['time_since_start_s'].min():.1f}s to {data['time_since_start_s'].max():.1f}s")

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

# Sglang metrics already synchronized above

# Already converted above

# Create metrics plot with 7 subplots
fig, axes = plt.subplots(7, 1, figsize=(12, 24), sharex=True)

# 1. TBT plot
for kind, group in df.groupby("kind"):
    group_sorted = group.sort_values("time_since_start_s")
    rolling_tbt = group_sorted["avg_tbt"].rolling(ROLLING_WINDOW).mean()
    axes[0].plot(group_sorted["time_since_start_s"], rolling_tbt, label=kind)
axes[0].set_ylabel("TBT (seconds)")
axes[0].legend(title="Kind")
axes[0].grid(True, alpha=0.3)

# 2. TTFT plot
for kind, group in df.groupby("kind"):
    group_sorted = group.sort_values("time_since_start_s")
    rolling_ttft = group_sorted["ttft"].rolling(ROLLING_WINDOW).mean()
    axes[1].plot(group_sorted["time_since_start_s"], rolling_ttft, label=kind)
axes[1].set_ylabel("TTFT (seconds)")
axes[1].legend(title="Kind")
axes[1].grid(True, alpha=0.3)

# Plot Prometheus metrics
metric_names = ['token_usage', 'throughput', 'running_reqs', 'queue_reqs']
for idx, metric_name in enumerate(metric_names):
    if metric_name in metrics_data:
        data = metrics_data[metric_name]
        axes[idx + 2].plot(
            data['time_since_start_s'],
            data['value'].rolling(ROLLING_WINDOW).mean(),
            label=f"{metric_name} ({METRICS_WINDOW} avg)"
        )
        axes[idx + 2].set_ylabel(metric_name.replace('_', ' ').title())
        axes[idx + 2].legend()
        axes[idx + 2].grid(True, alpha=0.3)

# 7. GPU Memory plot
axes[6].plot(gpu_df["time_since_start_s"], gpu_df["mem_used_MiB"], label='GPU Memory', alpha=0.6)
axes[6].set_ylabel("GPU Memory (MiB)")
axes[6].set_xlabel("Time since start (seconds)")
axes[6].legend()
axes[6].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(METRICS_COMBINED_OUT, dpi=300, bbox_inches='tight')
plt.close()

print("\nData ranges after synchronization:")
print(f"Latency time range: {df['time_since_start_s'].min():.1f}s to {df['time_since_start_s'].max():.1f}s")
for metric_name in metrics_data:
    data = metrics_data[metric_name]
    print(f"{metric_name} range: {data['time_since_start_s'].min():.1f}s to {data['time_since_start_s'].max():.1f}s")
