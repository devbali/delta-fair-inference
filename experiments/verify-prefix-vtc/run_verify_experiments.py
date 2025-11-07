import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = ROOT / "delta-fair-inference"
COMMANDS_FILE = Path(__file__).with_name("commands.txt")
PREFIX_VTC_DIR = REPO_ROOT / "Prefix-VTC"
DEFAULT_CLIENT_DIR = PREFIX_VTC_DIR / "benchmark" / "llm_judge"
VENV_DIR = ROOT / "prefix-vtc-venv"
PYTHON_EXECUTABLE = VENV_DIR / "bin" / "python"
ENV_FILE = REPO_ROOT / ".env.dev"


def ensure_pythonpath() -> None:
    prefix_vtc_python = PREFIX_VTC_DIR / "python"
    current = os.environ.get("PYTHONPATH", "")
    entries = [str(prefix_vtc_python)]
    if current:
        entries.append(current)
    os.environ["PYTHONPATH"] = os.pathsep.join(entries)


def parse_commands_file() -> List[Dict[str, str]]:
    experiments: List[Dict[str, str]] = []
    state = "expect_name"
    current: Dict[str, str] = {}

    for raw_line in COMMANDS_FILE.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if state == "expect_name":
            current = {"name": line}
            state = "expect_server_label"
            continue

        if state == "expect_server_label":
            if line.lower() != "server":
                raise ValueError(f"Expected 'Server' after experiment name, found '{line}'")
            state = "expect_server_command"
            continue

        if state == "expect_server_command":
            current["server_command"] = raw_line.strip()
            state = "expect_client_label"
            continue

        if state == "expect_client_label":
            if not line.lower().startswith("client"):
                raise ValueError(f"Expected 'Client' section, found '{line}'")
            dir_match = re.search(r"\(dir\s+([^)]+)\)", raw_line, flags=re.IGNORECASE)
            if dir_match:
                current["client_dir"] = dir_match.group(1).strip()
            state = "expect_client_command"
            continue

        if state == "expect_client_command":
            current["client_command"] = raw_line.strip()
            experiments.append(current)
            state = "expect_name"
            continue

        raise RuntimeError(f"Unhandled parser state '{state}'")

    if state != "expect_name":
        raise ValueError("commands.txt ended unexpectedly; check the file formatting.")

    return experiments


def replace_python_executable(tokens: List[str]) -> None:
    if tokens and tokens[0] in ("python", "python3"):
        tokens[0] = str(PYTHON_EXECUTABLE)


def consume_redirection(tokens: List[str]) -> Tuple[List[str], Optional[Path]]:
    clean_tokens: List[str] = []
    stdout_path: Optional[Path] = None

    for token in tokens:
        if token.startswith(">"):
            output_target = token[1:]
            stdout_path = Path(os.path.expandvars(os.path.expanduser(output_target)))
            continue
        clean_tokens.append(token)

    return clean_tokens, stdout_path


def format_for_launch(tokens: List[str]) -> str:
    return " ".join(shlex.quote(token) for token in tokens)


def load_env_file(env: Dict[str, str], path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Environment file not found: {path}")

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        env[key] = value

    print(f"[env] loaded variables from {path}")


def run_client(experiment: Dict[str, str], env: Dict[str, str]) -> None:
    client_tokens = shlex.split(experiment["client_command"], posix=True)
    client_tokens, stdout_path = consume_redirection(client_tokens)
    replace_python_executable(client_tokens)

    client_dir = experiment.get("client_dir")
    if client_dir:
        client_cwd = Path(os.path.expandvars(os.path.expanduser(client_dir)))
    else:
        client_cwd = DEFAULT_CLIENT_DIR

    if stdout_path is None:
        stdout_path = Path(__file__).with_name(f"{experiment['name']}.log")

    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[client] running {experiment['name']} -> {stdout_path}")
    with stdout_path.open("w") as out_file:
        subprocess.run(
            client_tokens,
            cwd=client_cwd,
            env=env,
            check=True,
            stdout=out_file,
            stderr=subprocess.STDOUT,
        )


def extract_port(tokens: List[str], default: int = 30000) -> int:
    for idx, token in enumerate(tokens):
        if token == "--port" and idx + 1 < len(tokens):
            return int(tokens[idx + 1])
        if token.startswith("--port="):
            return int(token.split("=", 1)[1])
    return default


def probe_server(port: int, server_process: subprocess.Popen, timeout: float = 600.0, interval: float = 2.0) -> None:
    """Repeatedly send lightweight /generate requests until one succeeds."""
    url = f"http://localhost:{port}/generate"
    payload = {
        "text": "Ping from run_verify_experiments",
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
    }
    start_time = time.time()

    while True:
        if server_process.poll() is not None:
            raise RuntimeError(
                "Server process terminated before becoming ready; inspect stderr for details."
            )
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.ok:
                print(f"[server] probe succeeded on port {port}")
                return
        except requests.RequestException:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Server did not respond successfully within {timeout} seconds.")
        time.sleep(interval)


def run_experiment(experiment: Dict[str, str], env: Dict[str, str]) -> None:
    server_tokens = shlex.split(experiment["server_command"], posix=True)
    replace_python_executable(server_tokens)
    port = extract_port(server_tokens)

    print(f"[server] starting {experiment['name']} on port {port}")
    server_process = subprocess.Popen(
        server_tokens,
        cwd=REPO_ROOT,
        env=env,
    )

    try:
        probe_server(port, server_process)
        run_client(experiment, env)
    finally:
        if server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_process.kill()
        print(f"[server] finished {experiment['name']}")


def main() -> None:
    ensure_pythonpath()
    if not PYTHON_EXECUTABLE.exists():
        raise FileNotFoundError(f"Cannot locate virtual environment python at {PYTHON_EXECUTABLE}")

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(PYTHON_EXECUTABLE.parent), env.get("PATH", "")])
    env["VIRTUAL_ENV"] = str(VENV_DIR)
    env.pop("PYTHONHOME", None)
    load_env_file(env, ENV_FILE)

    experiments = parse_commands_file()

    for experiment in experiments:
        run_experiment(experiment, env)


if __name__ == "__main__":
    main()
