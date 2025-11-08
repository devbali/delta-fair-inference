"""Runner for the 11-6 ablation study client/server pairs."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


load_env_file(REPO_ROOT / ".env.dev")
CLIENT = THIS_DIR / "client.py"
SERVER_WRAPPER = THIS_DIR / "server.py"
SERVER_PORT = 30000
SERVER_START_TIMEOUT = 180
VISIBLE_DEVICE = os.environ.get("ABLATION_VISIBLE_DEVICE", "0")
RESULTS_ROOT = THIS_DIR / "results"
DOCKER_IMAGE = os.environ.get("ABLATION_DOCKER_IMAGE", "ubuntu:24.04")
DOCKER_RUNTIME = os.environ.get("ABLATION_DOCKER_RUNTIME", "nvidia")
DOCKER_GPUS = os.environ.get("ABLATION_DOCKER_GPUS", VISIBLE_DEVICE)
HOST_PYTHON_EXECUTABLE = Path(os.environ.get("ABLATION_PYTHON_EXECUTABLE", sys.executable))
IS_ROOT = os.geteuid() == 0
PYTHON_EXECUTABLE = str(HOST_PYTHON_EXECUTABLE)
MODEL_ID = os.environ.get("ABLATION_MODEL_ID", "meta-llama/Llama-3.1-8B")


def detect_hf_cache_dir() -> Path:
    candidates = []
    env_dir = os.environ.get("HF_HOME")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path.home() / ".cache" / "huggingface")
    for user_key in ("SUDO_USER", "LOGNAME", "USER"):
        user = os.environ.get(user_key)
        if user:
            candidates.append(Path("/home") / user / ".cache" / "huggingface")
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            unique_candidates.append(path)
            seen.add(path)
    for path in unique_candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            continue
        os.environ.setdefault("HF_HOME", str(path))
        return path
    default = unique_candidates[0]
    default.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(default))
    return default


HF_CACHE_DIR = detect_hf_cache_dir()
HF_TOKEN_KEYS = [
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_API_TOKEN",
    "HF_TOKEN",
]
DOCKER_LABEL_KEY = "ablation_run"
DOCKER_LABEL_VALUE = "1"
COMPUTE_SHARE_DEVICE_MAP: dict[str, str] = {}
_share_map_raw = os.environ.get("ABLATION_COMPUTE_SHARE_DEVICES")
if _share_map_raw:
    for entry in _share_map_raw.split(","):
        if "=" not in entry:
            continue
        k, v = entry.split("=", 1)
        key = k.strip()
        value = v.strip()
        if key and value:
            COMPUTE_SHARE_DEVICE_MAP[key] = value

_default_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY") or os.environ.get(
    "ABLATION_MPS_PIPE_DIR", "/tmp/ablation-mps-pipe"
)
_default_log = os.environ.get("CUDA_MPS_LOG_DIRECTORY") or os.environ.get(
    "ABLATION_MPS_LOG_DIR", "/tmp/ablation-mps-log"
)
MPS_PIPE_DIR = Path(_default_pipe)
MPS_LOG_DIR = Path(_default_log)
os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", str(MPS_PIPE_DIR))
os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", str(MPS_LOG_DIR))
MPS_ENABLED = os.environ.get("ABLATION_ENABLE_MPS", "1").lower() not in {"0", "false", "no"}
MPS_CONTROL_BIN = shutil.which("nvidia-cuda-mps-control")
POWER_SCALING_ENABLED = os.environ.get("ABLATION_ENABLE_POWER_SCALING", "1").lower() not in {"0", "false", "no"}
POWER_LIMIT_CACHE: Dict[str, float] = {}
POWER_DEFAULT_LIMIT_CACHE: Dict[str, float] = {}
MIN_POWER_WATTS = float(os.environ.get("ABLATION_MIN_POWER_WATTS", "50"))
MAX_POWER_WATTS_OVERRIDE = os.environ.get("ABLATION_MAX_POWER_WATTS")


def has_existing_results(group: str, name: str) -> bool:
    exp_dir = RESULTS_ROOT / group / name
    if not exp_dir.exists():
        return False
    return any(child.is_dir() for child in exp_dir.iterdir())

ARTIFACT_PATTERNS = [
    "gpu_metrics_*.csv",
    "prometheus_metrics_*.csv",
    "token_usage_*.csv",
    "latencies_*.csv",
]
STATIC_ARTIFACTS = [
    "sglang_log_decode.csv",
]


def compute_share_pct(value: float) -> int:
    pct = float(value)
    if pct <= 1:
        pct *= 100
    return int(max(1.0, min(100.0, pct)))


def resolve_model_path(model_id: str) -> str:
    override = os.environ.get("ABLATION_LOCAL_MODEL_PATH")
    if override:
        return override

    cache_root = HF_CACHE_DIR / "hub"
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    ref_file = repo_dir / "refs" / "main"
    if ref_file.exists():
        snapshot_hash = ref_file.read_text().strip()
        snapshot_dir = repo_dir / "snapshots" / snapshot_hash
        if snapshot_dir.exists():
            return str(snapshot_dir)
    return model_id


MODEL_PATH = resolve_model_path(MODEL_ID)
MODEL_IS_LOCAL = Path(MODEL_PATH).exists()


def _maybe_sudo(cmd: list[str], sudo: bool) -> list[str]:
    if sudo and not IS_ROOT:
        return ["sudo"] + cmd
    return cmd


def run_cmd(
    cmd: list[str],
    *,
    sudo: bool = False,
    capture_output: bool = False,
    text: bool = True,
    check: bool = True,
    env: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    actual_cmd = _maybe_sudo(cmd, sudo)
    return subprocess.run(
        actual_cmd,
        capture_output=capture_output,
        text=text,
        check=check,
        env=env,
    )


def canonical_share_key(value: float) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num.is_integer():
        return str(int(num))
    return str(num)


def resolve_device_for_share(value: float) -> str | None:
    key = canonical_share_key(value)
    device = COMPUTE_SHARE_DEVICE_MAP.get(key)
    if device:
        return device
    return None


def kill_gpu_processes(devices: str | None = None) -> None:
    target_devices = devices or VISIBLE_DEVICE
    pids_to_kill: set[int] = set()
    for dev in target_devices.split(","):
        dev = dev.strip()
        if not dev:
            continue
        cmd = [
            "nvidia-smi",
            "-i",
            dev,
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ]
        try:
            result = run_cmd(cmd, capture_output=True, check=False)
        except FileNotFoundError:
            print("nvidia-smi not found; skipping GPU cleanup.")
            return
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or not re.fullmatch(r"\d+", line):
                continue
            pids_to_kill.add(int(line))

    for pid in pids_to_kill:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"Killed GPU process PID {pid}")
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            print(f"Failed to kill PID {pid}: {exc}")


def stop_mps_daemon(timeout: float = 3.0) -> None:
    if not MPS_CONTROL_BIN:
        return
    try:
        proc = subprocess.Popen(
            _maybe_sudo([MPS_CONTROL_BIN, "-s"], sudo=not IS_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait(timeout=timeout)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    except subprocess.TimeoutExpired:
        proc.kill()

    # If daemon is still alive, use pkill as a last resort
    run_cmd(
        ["pkill", "-f", "nvidia-cuda-mps-control"],
        sudo=not IS_ROOT,
        check=False,
        capture_output=True,
    )


def start_mps_daemon() -> bool:
    if not MPS_CONTROL_BIN:
        return False
    env = os.environ.copy()
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(MPS_PIPE_DIR)
    env["CUDA_MPS_LOG_DIRECTORY"] = str(MPS_LOG_DIR)
    try:
        result = run_cmd(
            [MPS_CONTROL_BIN, "-d"],
            sudo=not IS_ROOT,
            check=False,
            capture_output=True,
            env=env,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def ensure_mps_daemon() -> None:
    if not (MPS_ENABLED and MPS_CONTROL_BIN):
        return
    for directory in (MPS_PIPE_DIR, MPS_LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    stop_mps_daemon()
    if start_mps_daemon():
        atexit.register(stop_mps_daemon)
        print(
            f"Restarted NVIDIA MPS control daemon with pipe={MPS_PIPE_DIR} log={MPS_LOG_DIR}"
        )


def _parse_power_value(raw: str) -> float:
    token = raw.strip().split()[0]
    return float(token)


def _query_power_value(device: str, field: str) -> float:
    result = run_cmd(
        [
            "nvidia-smi",
            "-i",
            device,
            f"--query-gpu={field}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        sudo=not IS_ROOT,
        check=True,
    )
    return float(result.stdout.strip().splitlines()[0])


def set_power_limit(device: str, watts: float) -> None:
    run_cmd(
        ["nvidia-smi", "-i", device, "-pl", f"{watts:.0f}"],
        sudo=not IS_ROOT,
        check=True,
    )


def apply_power_limit(device: str, share_pct: float) -> Callable[[], None]:
    if not POWER_SCALING_ENABLED:
        return lambda: None
    try:
        device_index = str(int(device))
    except ValueError:
        # Non-integer device identifiers (e.g., MIG) unsupported for power scaling
        return lambda: None
    share_pct = float(share_pct)
    if share_pct >= 99.9:
        return lambda: None

    try:
        if device_index not in POWER_DEFAULT_LIMIT_CACHE:
            POWER_DEFAULT_LIMIT_CACHE[device_index] = _query_power_value(device_index, "power.limit")
        default_limit = POWER_DEFAULT_LIMIT_CACHE[device_index]

        if MAX_POWER_WATTS_OVERRIDE is not None:
            base_limit = float(MAX_POWER_WATTS_OVERRIDE)
        else:
            base_limit = _query_power_value(device_index, "power.max_limit")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[warn] Unable to query power limits for GPU {device_index}: {exc}")
        return lambda: None

    new_limit = max(MIN_POWER_WATTS, base_limit * share_pct / 100.0)
    if new_limit >= default_limit - 1:
        return lambda: None

    print(f"Setting GPU {device_index} power limit to {new_limit:.1f} W for compute share {share_pct}%")
    set_power_limit(device_index, new_limit)

    def restore() -> None:
        print(f"Restoring GPU {device_index} power limit to {default_limit:.1f} W")
        set_power_limit(device_index, default_limit)

    return restore


def cleanup_docker_containers() -> None:
    if not DOCKER_IMAGE:
        return
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-q",
                "--filter",
                f"label={DOCKER_LABEL_KEY}={DOCKER_LABEL_VALUE}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return
    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for container_id in ids:
        subprocess.run(["docker", "rm", "-f", container_id], check=False)


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


def default_server_cmd(device: str | None = None) -> list[str]:
    base_cmd = [
        PYTHON_EXECUTABLE,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
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
        "--dtype",
        "float16",
    ]
    return dockerize_server_cmd(base_cmd, docker_device=device)


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
    share_device = resolve_device_for_share(compute_share) or VISIBLE_DEVICE
    base_cmd = [
        PYTHON_EXECUTABLE,
        str(SERVER_WRAPPER),
        "--visible-device",
        share_device,
        "--compute-share",
        str(compute_share),
        "--model-path",
        MODEL_PATH,
        "--port",
        str(SERVER_PORT),
    ]
    cmd = dockerize_server_cmd(base_cmd, compute_share, docker_device=share_device)
    return cmd


def dockerize_server_cmd(
    cmd: list[str],
    compute_share: float | None = None,
    docker_device: str | None = None,
) -> list[str]:
    if not DOCKER_IMAGE:
        return cmd

    device = docker_device or DOCKER_GPUS

    env_flags = [
        "-e",
        f"NVIDIA_VISIBLE_DEVICES={device}",
        "-e",
        f"CUDA_VISIBLE_DEVICES={device}",
        "-e",
        f"HF_HOME={HF_CACHE_DIR}",
    ]
    if MODEL_IS_LOCAL:
        env_flags.extend(
            [
                "-e",
                "HF_HUB_OFFLINE=1",
                "-e",
                "TRANSFORMERS_OFFLINE=1",
                "-e",
                "HF_DATASETS_OFFLINE=1",
            ]
        )
    for env_var in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"):
        value = os.environ.get(env_var)
        if value:
            env_flags.extend(["-e", f"{env_var}={value}"])
    for token_key in HF_TOKEN_KEYS:
        token_value = os.environ.get(token_key)
        if token_value:
            env_flags.extend(["-e", f"{token_key}={token_value}"])
    if compute_share is not None:
        env_flags.extend(
            [
                "-e",
                f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={compute_share_pct(compute_share)}",
            ]
        )

    host_cmd = f"cd {shlex.quote(str(REPO_ROOT))} && {shlex.join(cmd)}"

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--runtime",
        DOCKER_RUNTIME,
        "--gpus",
        f"device={device}",
        "--label",
        f"{DOCKER_LABEL_KEY}={DOCKER_LABEL_VALUE}",
        "-v",
        "/:/hostroot",
    ]
    docker_cmd.extend(env_flags)
    docker_cmd.append(DOCKER_IMAGE)
    docker_cmd.extend(
        [
            "chroot",
            "/hostroot",
            "/bin/bash",
            "-lc",
            host_cmd,
        ]
    )
    return docker_cmd


EXPERIMENTS = [
    # Ablation 1: prompt tokens sweep
    {
        "name": "abl1_prompt_050",
        "group": "ablation1",
        "server_cmd": default_server_cmd(VISIBLE_DEVICE),
        "client_cmd": client_cmd(prompt_words=50, completion_tokens=2000, request_rate=1, duration=120, max_concurrent=800),
    },
    {
        "name": "abl1_prompt_200",
        "group": "ablation1",
        "server_cmd": default_server_cmd(VISIBLE_DEVICE),
        "client_cmd": client_cmd(prompt_words=200, completion_tokens=2000, request_rate=1, duration=120, max_concurrent=800),
    },
    {
        "name": "abl1_prompt_500",
        "group": "ablation1",
        "server_cmd": default_server_cmd(VISIBLE_DEVICE),
        "client_cmd": client_cmd(prompt_words=500, completion_tokens=2000, request_rate=1, duration=120, max_concurrent=800),
    },
    {
        "name": "abl1_prompt_800",
        "group": "ablation1",
        "server_cmd": default_server_cmd(VISIBLE_DEVICE),
        "client_cmd": client_cmd(prompt_words=800, completion_tokens=2000, request_rate=1, duration=120, max_concurrent=800),
    },
    {
        "name": "abl1_prompt_2000",
        "group": "ablation1",
        "server_cmd": default_server_cmd(VISIBLE_DEVICE),
        "client_cmd": client_cmd(prompt_words=2000, completion_tokens=2000, request_rate=1, duration=120, max_concurrent=800),
    },

    # Ablation 2: request rate sweep with tiny prompts
    {
        "name": "abl2_rate_1",
        "group": "ablation2",
        "server_cmd": default_server_cmd(VISIBLE_DEVICE),
        "client_cmd": client_cmd(prompt_words=10, completion_tokens=1800, request_rate=1, duration=120, max_concurrent=800),
    },
    {
        "name": "abl2_rate_2",
        "group": "ablation2",
        "server_cmd": default_server_cmd(VISIBLE_DEVICE),
        "client_cmd": client_cmd(prompt_words=10, completion_tokens=1800, request_rate=2, duration=120, max_concurrent=800),
    },
    {
        "name": "abl2_rate_6",
        "group": "ablation2",
        "server_cmd": default_server_cmd(VISIBLE_DEVICE),
        "client_cmd": client_cmd(prompt_words=10, completion_tokens=1800, request_rate=6, duration=120, max_concurrent=800),
    },
    {
        "name": "abl2_rate_12",
        "group": "ablation2",
        "server_cmd": default_server_cmd(VISIBLE_DEVICE),
        "client_cmd": client_cmd(prompt_words=10, completion_tokens=1800, request_rate=12, duration=120, max_concurrent=800),
    },
    # Ablation 3: compute share sweep (same client command)
    {
        "name": "abl3_share_100",
        "group": "ablation3",
        "server_cmd": partitioned_server_cmd(100),
        "compute_share": 100,
        "device": resolve_device_for_share(100) or VISIBLE_DEVICE,
        "client_cmd": client_cmd(prompt_words=10, completion_tokens=1000, request_rate=1, duration=120, max_concurrent=800),
    },
    {
        "name": "abl3_share_60",
        "group": "ablation3",
        "server_cmd": partitioned_server_cmd(60),
        "compute_share": 60,
        "device": resolve_device_for_share(60) or VISIBLE_DEVICE,
        "client_cmd": client_cmd(prompt_words=10, completion_tokens=1000, request_rate=1, duration=120, max_concurrent=800),
    },
    {
        "name": "abl3_share_30",
        "group": "ablation3",
        "server_cmd": partitioned_server_cmd(30),
        "compute_share": 30,
        "device": resolve_device_for_share(30) or VISIBLE_DEVICE,
        "client_cmd": client_cmd(prompt_words=10, completion_tokens=1000, request_rate=1, duration=120, max_concurrent=800),
    },
    {
        "name": "abl3_share_10",
        "group": "ablation3",
        "server_cmd": partitioned_server_cmd(10),
        "compute_share": 10,
        "device": resolve_device_for_share(10) or VISIBLE_DEVICE,
        "client_cmd": client_cmd(prompt_words=10, completion_tokens=1000, request_rate=1, duration=120, max_concurrent=400),
    },
    # {
    #     "name": "abl3_share_2",
    #     "group": "ablation3",
    #     "server_cmd": partitioned_server_cmd(2),
    #     "client_cmd": client_cmd(prompt_words=10, completion_tokens=1000, request_rate=1, duration=120, max_concurrent=400),
    # }
]

for _exp in EXPERIMENTS:
    _exp.setdefault("device", VISIBLE_DEVICE)
    _exp.setdefault("compute_share", 100.0)


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
    device = str(exp.get("device", VISIBLE_DEVICE))
    share_pct = float(exp.get("compute_share", 100.0))
    result_dir = ensure_result_dir(group, name)
    rel_result = result_dir.relative_to(REPO_ROOT)
    print(f"\n=== Starting {name} (results → {rel_result}) ===")
    server_cmd: list[str] = exp["server_cmd"]  # type: ignore[assignment]
    client_cmd_args: list[str] = exp["client_cmd"]  # type: ignore[assignment]
    before_snapshot = artifact_snapshot()
    restore_power = apply_power_limit(device, share_pct)
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
        restore_power()
        cleanup_docker_containers()
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
    ensure_mps_daemon()
    for exp in selected:
        if has_existing_results(exp["group"], exp["name"]):  # type: ignore[index]
            print(f"Skipping {exp['name']} – results already exist.")
            continue
        ran = False
        try:
            ran = True
            run_pair(exp)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Experiment {exp['name']} failed: {exc}")
            if ran:
                kill_gpu_processes(str(exp.get("device", VISIBLE_DEVICE)))
            return 1
        kill_gpu_processes(str(exp.get("device", VISIBLE_DEVICE)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
