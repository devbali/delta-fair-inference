"""Utility launcher for ablation servers with fixed GPU partitions.

This wrapper mirrors how the NVIDIA container runtime constrains workloads:
it pins the server process to a specific (MIG) visible device and optionally
reduces its CUDA MPS slice by setting `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`.

Example:
    python server.py --visible-device MIG-GPU-abcdef/1/0 --compute-share 60
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_compute_share(value: float) -> int:
    """Return an integer percentage within [1, 100]."""
    pct = float(value)
    if pct <= 1:
        pct *= 100
    pct = max(1.0, min(100.0, pct))
    return int(pct)


def build_server_cmd(args: argparse.Namespace) -> list[str]:
    """Compose the underlying sglang server command."""
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--enable-eviction",
        "--enable-iterative-eviction",
        "--port",
        str(args.port),
        "--mem-frac",
        str(args.mem_frac),
        "--schedule-po",
        args.schedule_policy,
        "--load-ba",
        args.load_batcher,
        "--Q",
        str(args.queue_limit),
        "--global-Q",
        str(args.global_queue_limit),
        "--max-total",
        str(args.max_total_tokens),
        "--max-prefill",
        str(args.max_prefill_tokens),
        "--dtype",
        args.dtype,
    ]
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def run_server(cmd: list[str], env: dict[str, str], cwd: Path) -> int:
    """Launch the server and forward its exit code."""
    print("Launching server command:\n  " + " ".join(cmd))
    process = subprocess.Popen(cmd, env=env, cwd=cwd)
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the ablation server with a fixed GPU partition.")
    parser.add_argument("--visible-device", default="0",
                        help="CUDA or MIG identifier passed to NVIDIA_VISIBLE_DEVICES (default: %(default)s)")
    parser.add_argument("--compute-share", type=float, default=100.0,
                        help="Fraction or percentage (<=1 => fraction) for CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.")
    parser.add_argument("--model-path", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-frac", type=float, default=0.7)
    parser.add_argument("--schedule-policy", default="lpm_mdrr")
    parser.add_argument("--load-batcher", default="hdoubleq")
    parser.add_argument("--queue-limit", type=int, default=50000)
    parser.add_argument("--global-queue-limit", type=int, default=100000)
    parser.add_argument("--max-total-tokens", type=int, default=20000)
    parser.add_argument("--max-prefill-tokens", type=int, default=20000)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER,
                        help="Additional arguments appended verbatim to the server command (prefix with --).")

    args = parser.parse_args()
    cmd = build_server_cmd(args)

    env = os.environ.copy()
    env.update({
        "NVIDIA_VISIBLE_DEVICES": args.visible_device,
        "CUDA_VISIBLE_DEVICES": args.visible_device,
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(parse_compute_share(args.compute_share)),
        "NVIDIA_DRIVER_CAPABILITIES": env.get("NVIDIA_DRIVER_CAPABILITIES", "compute,utility"),
    })

    repo_root = Path(__file__).resolve().parents[2]
    env.setdefault("PYTHONPATH", str(repo_root))
    return run_server(cmd, env, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
