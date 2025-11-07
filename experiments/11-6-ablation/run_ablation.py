"""Runner for the 11-6 ablation study client/server pairs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
CLIENT = THIS_DIR / "client.py"
SERVER_WRAPPER = THIS_DIR / "server.py"
SERVER_PORT = 30000
SERVER_START_TIMEOUT = 180
VISIBLE_DEVICE = os.environ.get("ABLATION_VISIBLE_DEVICE", "0")
RESULTS_ROOT = THIS_DIR / "results"

ARTIFACT_PATTERNS = [
    "gpu_metrics_*.csv",
    "prometheus_metrics_*.csv",
    "token_usage_*.csv",
    "latencies_*.csv",
]
STATIC_ARTIFACTS = [
    "sglang_log_decode.csv",
]


def artifact_snapshot() -> dict[Path, int]:
    snapshot: dict[Path, int] = {}

    def _record(path: Path) -> None:
        try:
            snapshot[path.resolve()] = path.stat().st_mtime_ns
        except FileNotFoundError:
            return

    for pattern in ARTIFACT_PATTERNS:
        for path in REPO_ROOT.glob(pattern):
            _record(path)
    for rel in STATIC_ARTIFACTS:
        path = REPO_ROOT / rel
        if path.exists():
            _record(path)
    return snapshot


def ensure_result_dir(group: str, name: str) -> Path:
    base = RESULTS_ROOT / group / name
    base.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dest = base / timestamp
    counter = 1
    while dest.exists():
        dest = base / f"{timestamp}_{counter:02d}"
        counter += 1
    dest.mkdir(parents=True, exist_ok=False)
    return dest


def move_with_unique_name(src: Path, dest_dir: Path) -> Path:
    target = dest_dir / src.name
    counter = 1
    while target.exists():
        target = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    shutil.move(str(src), target)
    return target


def collect_artifacts(before: dict[Path, int], dest_dir: Path) -> list[Path]:
    after = artifact_snapshot()
    changed: list[Path] = []
    for path, mtime in after.items():
        previous = before.get(path)
        if previous is None or mtime > previous:
            changed.append(path)

    collected: list[Path] = []
    for path in changed:
        if not path.exists():
            continue
        collected.append(move_with_unique_name(path, dest_dir))
    return collected


def write_metadata(dest_dir: Path, exp: dict[str, object], server_cmd: list[str], client_cmd: list[str],
                   success: bool, error: str | None, collected: list[Path]) -> None:
    metadata = {
        "experiment": exp["name"],
        "group": exp["group"],
        "server_cmd": server_cmd,
        "client_cmd": client_cmd,
        "success": success,
        "error": error,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "artifacts": [path.name for path in collected],
    }
    (dest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def default_server_cmd() -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        "meta-llama/Llama-3.1-8B",
        "--enable-eviction",
        "--enable-iterative-eviction",
        "--port",
        str(SERVER_PORT),
        "--mem-frac",
        "0.7",
        "--schedule-po",
        "lpm_mdrr",
        "--load-ba",
        "hdoubleq",
        "--Q",
        "50000",
        "--global-Q",
        "100000",
        "--max-total",
        "20000",
        "--max-prefill",
        "20000",
        "--dtype",
        "float16",
    ]


def client_cmd(**kwargs: str) -> list[str]:
    cmd = [
        sys.executable,
        str(CLIENT),
    ]
    for key, value in kwargs.items():
        flag = f"--{key.replace('_', '-')}"
        cmd.extend([flag, str(value)])
    return cmd


def partitioned_server_cmd(compute_share: float) -> list[str]:
    return [
        sys.executable,
        str(SERVER_WRAPPER),
        "--visible-device",
        VISIBLE_DEVICE,
        "--compute-share",
        str(compute_share),
        "--port",
        str(SERVER_PORT),
    ]


EXPERIMENTS = [
    # Ablation 1: prompt tokens sweep
    {
        "name": "abl1_prompt_050",
        "group": "ablation1",
        "server_cmd": default_server_cmd(),
        "client_cmd": client_cmd(prompt_words=50, completion_tokens=2000, request_rate=1.5, duration=180, max_concurrent=300),
    },
    {
        "name": "abl1_prompt_200",
        "group": "ablation1",
        "server_cmd": default_server_cmd(),
        "client_cmd": client_cmd(prompt_words=200, completion_tokens=2000, request_rate=1.5, duration=180, max_concurrent=300),
    },
    {
        "name": "abl1_prompt_500",
        "group": "ablation1",
        "server_cmd": default_server_cmd(),
        "client_cmd": client_cmd(prompt_words=500, completion_tokens=2000, request_rate=1.5, duration=180, max_concurrent=300),
    },
    {
        "name": "abl1_prompt_800",
        "group": "ablation1",
        "server_cmd": default_server_cmd(),
        "client_cmd": client_cmd(prompt_words=800, completion_tokens=2000, request_rate=1.5, duration=180, max_concurrent=300),
    },
    # Ablation 2: request rate sweep with tiny prompts
    {
        "name": "abl2_rate_2",
        "group": "ablation2",
        "server_cmd": default_server_cmd(),
        "client_cmd": client_cmd(prompt_words=10, completion_tokens=1800, request_rate=2, duration=180, max_concurrent=400),
    },
    {
        "name": "abl2_rate_6",
        "group": "ablation2",
        "server_cmd": default_server_cmd(),
        "client_cmd": client_cmd(prompt_words=10, completion_tokens=1800, request_rate=6, duration=180, max_concurrent=600),
    },
    {
        "name": "abl2_rate_12",
        "group": "ablation2",
        "server_cmd": default_server_cmd(),
        "client_cmd": client_cmd(prompt_words=10, completion_tokens=1800, request_rate=12, duration=180, max_concurrent=900),
    },
    # Ablation 3: compute share sweep (same client command)
    {
        "name": "abl3_share_100",
        "group": "ablation3",
        "server_cmd": partitioned_server_cmd(100),
        "client_cmd": client_cmd(prompt_words=600, completion_tokens=2500, request_rate=2, duration=240, max_concurrent=400),
    },
    {
        "name": "abl3_share_60",
        "group": "ablation3",
        "server_cmd": partitioned_server_cmd(60),
        "client_cmd": client_cmd(prompt_words=600, completion_tokens=2500, request_rate=2, duration=240, max_concurrent=400),
    },
    {
        "name": "abl3_share_30",
        "group": "ablation3",
        "server_cmd": partitioned_server_cmd(30),
        "client_cmd": client_cmd(prompt_words=600, completion_tokens=2500, request_rate=2, duration=240, max_concurrent=400),
    },
]


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: int = SERVER_START_TIMEOUT) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            try:
                sock.connect((host, port))
                return
            except OSError:
                time.sleep(1)
    raise TimeoutError(f"Server did not open {host}:{port} within {timeout}s")


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_pair(exp: dict[str, object]) -> None:
    name = exp["name"]
    group = exp["group"]
    result_dir = ensure_result_dir(group, name)
    rel_result = result_dir.relative_to(REPO_ROOT)
    print(f"\n=== Starting {name} (results → {rel_result}) ===")
    server_cmd: list[str] = exp["server_cmd"]  # type: ignore[assignment]
    client_cmd_args: list[str] = exp["client_cmd"]  # type: ignore[assignment]
    before_snapshot = artifact_snapshot()
    server = subprocess.Popen(server_cmd, cwd=REPO_ROOT)
    success = False
    error_msg: str | None = None
    collected: list[Path] = []
    try:
        wait_for_port(SERVER_PORT)
        subprocess.run(client_cmd_args, cwd=REPO_ROOT, check=True)
        success = True
    except Exception as exc:
        error_msg = str(exc)
        raise
    finally:
        stop_process(server)
        collected = collect_artifacts(before_snapshot, result_dir)
        write_metadata(result_dir, exp, server_cmd, client_cmd_args, success, error_msg, collected)
        print(f"Artifacts stored in {rel_result}")
    print(f"=== Finished {name} ===\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 11-6 ablation matrix.")
    parser.add_argument("--groups", nargs="*", choices=sorted({e["group"] for e in EXPERIMENTS}),
                        help="Subset of ablation groups to run (default: all).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [
        exp for exp in EXPERIMENTS
        if not args.groups or exp["group"] in args.groups
    ]
    if not selected:
        print("No experiments selected.")
        return 0
    for exp in selected:
        try:
            run_pair(exp)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Experiment {exp['name']} failed: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
