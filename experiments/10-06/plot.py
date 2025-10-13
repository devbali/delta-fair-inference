import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Settings ---
CSV_PATH = "/home/devbali/fairinf/10-06/latencies_20251009_031853.csv"
GPU_CSV_PATH = CSV_PATH.replace("latencies_", "gpu_metrics_")
PROM_CSV_PATH = CSV_PATH.replace("latencies_", "prometheus_metrics_")
SCATTER_OUT = "latency_per_token_over_time.png"
BARS_OUT = "token_averages_by_kind.png"
ROLLING_OUT = "rolling_avg_latency_per_token.png"
GPU_COMBINED_OUT = "latency_and_gpu_memory.png"
METRICS_COMBINED_OUT = "latency_and_metrics.png"

# Prometheus metric names to plot
PROM_METRICS = [
    'sglang:token_usage{engine_type="unified",model_name="qwen/qwen2.5-0.5b-instruct",pp_rank="0",tp_rank="0"}',
    'sglang:gen_throughput{engine_type="unified",model_name="qwen/qwen2.5-0.5b-instruct",pp_rank="0",tp_rank="0"}',
    'sglang:num_running_reqs{engine_type="unified",model_name="qwen/qwen2.5-0.5b-instruct",pp_rank="0",tp_rank="0"}'
]
METRIC_LABELS = ["Token Usage", "Generation Throughput", "Running Requests"]
ROLLING_WINDOW = 100  # rolling mean window in samples
CUT_FIRST_SECONDS = 10  # ignore first 10 seconds

# --- Load & prep ---
df = pd.read_csv(CSV_PATH)
df["timestamp_start_iso"] = pd.to_datetime(df["timestamp_start_iso"], utc=True)

# Print timestamp for debugging
print("Sample latency timestamp:", df["timestamp_start_iso"].iloc[0])

# Compute time since start (seconds)
t0 = df["timestamp_start_iso"].iloc[0]
df["time_since_start_s"] = (df["timestamp_start_iso"] - t0).dt.total_seconds()

# Cut out the first N seconds
df = df[df["time_since_start_s"] >= CUT_FIRST_SECONDS].copy()

# Latency per completion token (avoid divide-by-zero)
df["latency_per_completion_token"] = np.where(
    df["completion_tokens"] > 0,
    df["duration_s"] / df["completion_tokens"],
    np.nan
)
df["kind"] = df["kind"].astype(str)

# ======================================================
# 1) Scatter: latency/token vs. time (filtered)
# ======================================================
plt.figure(figsize=(10, 6))
for kind, group in df.groupby("kind"):
    plt.scatter(
        group["time_since_start_s"],
        group["latency_per_completion_token"],
        label=kind,
        s=20
    )

plt.xlabel("Time since start (seconds)")
plt.ylabel("Latency per completion token (s/token)")
plt.title("Latency per Completion Token Over Time Since Start (≥10 s)")
plt.legend(title="Kind")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCATTER_OUT, dpi=300)
plt.close()


# ======================================================
# 2) Grouped bars: averages with error bars (std deviation)
# ======================================================
metrics = ["prompt_tokens", "completion_tokens", "total_tokens"]
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

plt.xticks(x, ["Prompt tokens", "Completion tokens", "Total tokens"])
plt.ylabel("Average tokens")
plt.title("Average Token Counts by Kind (error bars = ±1 SD, ≥10 s)")
plt.legend(title="Kind")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(BARS_OUT, dpi=300)
plt.close()

# ======================================================
# 3) Rolling average (latency per token) by kind (filtered)
# ======================================================
plt.figure(figsize=(10, 6))
for kind, group in df.groupby("kind"):
    group_sorted = group.sort_values("time_since_start_s")
    rolling = group_sorted["latency_per_completion_token"].rolling(ROLLING_WINDOW).mean()
    plt.plot(group_sorted["time_since_start_s"], rolling, label=f"{kind} (rolling avg)")

plt.xlabel("Time since start (seconds)")
plt.ylabel(f"Rolling mean latency per completion token (s/token, window={ROLLING_WINDOW})")
plt.title(f"Rolling Average of Latency per Completion Token by Kind (≥{CUT_FIRST_SECONDS}s)")
plt.legend(title="Kind")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ROLLING_OUT, dpi=300)
plt.close()

# ======================================================
# 4) Combined Metrics Plot
# ======================================================
# Load GPU metrics
gpu_df = pd.read_csv(GPU_CSV_PATH)
gpu_df["timestamp"] = pd.to_datetime(gpu_df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
gpu_t0 = gpu_df["timestamp"].iloc[0]
gpu_df["time_since_start_s"] = (gpu_df["timestamp"] - gpu_t0).dt.total_seconds()

# Load Prometheus metrics
prom_df = pd.read_csv(PROM_CSV_PATH)
prom_df["timestamp"] = pd.to_datetime(prom_df["timestamp"])
prom_t0 = prom_df["timestamp"].iloc[0]
prom_df["time_since_start_s"] = (prom_df["timestamp"] - prom_t0).dt.total_seconds()

# Filter metrics and pivot the data
prom_data = {}
for metric in PROM_METRICS:
    metric_data = prom_df[prom_df["metric_name"] == metric].copy()
    metric_data["value"] = pd.to_numeric(metric_data["value"], errors="coerce")
    prom_data[metric] = metric_data

# Trim data to match time ranges
max_time = min(
    df["time_since_start_s"].max(),
    gpu_df["time_since_start_s"].max(),
    max(data["time_since_start_s"].max() for data in prom_data.values())
)

df = df[df["time_since_start_s"] <= max_time]
gpu_df = gpu_df[gpu_df["time_since_start_s"] <= max_time]
for metric in PROM_METRICS:
    prom_data[metric] = prom_data[metric][prom_data[metric]["time_since_start_s"] <= max_time]

# Filter out first N seconds
gpu_df = gpu_df[gpu_df["time_since_start_s"] >= CUT_FIRST_SECONDS]
for metric in PROM_METRICS:
    prom_data[metric] = prom_data[metric][prom_data[metric]["time_since_start_s"] >= CUT_FIRST_SECONDS]

# Create GPU memory and latency plot
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

# Plot latency rolling average on the primary y-axis
for kind, group in df.groupby("kind"):
    group_sorted = group.sort_values("time_since_start_s")
    rolling = group_sorted["latency_per_completion_token"].rolling(ROLLING_WINDOW).mean()
    ax1.plot(group_sorted["time_since_start_s"], rolling, label=f"{kind} (latency)", linestyle='-')

# Plot GPU memory usage on the secondary y-axis
memory_line = ax2.plot(gpu_df["time_since_start_s"], gpu_df["mem_used_MiB"], 
                      color='red', label='GPU Memory Used', alpha=0.6)

# Customize the plot
ax1.set_xlabel("Time since start (seconds)")
ax1.set_ylabel("Rolling mean latency per completion token (s/token)")
ax2.set_ylabel("GPU Memory Used (MiB)")

# Add legends for both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title("Latency and GPU Memory Usage Over Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(GPU_COMBINED_OUT, dpi=300)
plt.close()

# Create metrics plot with 4 subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
fig.suptitle("Latency and Metrics Over Time", fontsize=16, y=0.95)

# 1. Latency plot
for kind, group in df.groupby("kind"):
    group_sorted = group.sort_values("time_since_start_s")
    rolling = group_sorted["latency_per_completion_token"].rolling(ROLLING_WINDOW).mean()
    ax1.plot(group_sorted["time_since_start_s"], rolling, label=f"{kind} (latency)")
ax1.set_ylabel("Latency per completion\ntoken (s/token)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Token Usage plot
token_data = prom_data[PROM_METRICS[0]]
ax2.plot(token_data["time_since_start_s"], token_data["value"], label="Token Usage")
ax2.set_ylabel("Token Usage")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Generation Throughput plot
throughput_data = prom_data[PROM_METRICS[1]]
ax3.plot(throughput_data["time_since_start_s"], throughput_data["value"], label="Generation Throughput")
ax3.set_ylabel("Generation Throughput\n(tokens/sec)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Running Requests plot
requests_data = prom_data[PROM_METRICS[2]]
ax4.plot(requests_data["time_since_start_s"], requests_data["value"], label="Running Requests")
ax4.set_ylabel("Number of\nRunning Requests")
ax4.set_xlabel("Time since start (seconds)")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(METRICS_COMBINED_OUT, dpi=300, bbox_inches='tight')
plt.close()

print("\nData ranges after synchronization:")
print(f"Latency time range: {df['time_since_start_s'].min():.1f}s to {df['time_since_start_s'].max():.1f}s")
for metric, label in zip(PROM_METRICS, METRIC_LABELS):
    data = prom_data[metric]
    print(f"{label} time range: {data['time_since_start_s'].min():.1f}s to {data['time_since_start_s'].max():.1f}s")
