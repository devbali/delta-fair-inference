import threading
import queue
import time
import random
from datetime import datetime
import os
import csv
import subprocess

regular_latencies = []
bad_client_latencies = []
MAX_THREADS = 1000  # max concurrent threads
LAUNCH_INTERVAL = 0.08  # seconds between launches
TOTAL_TIME = 600
BAD_TASK_ID_START = 4000
IS_GOOD_TASK = lambda task_id: task_id % 20 < 15

RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
GPU_CSV_PATH = f"gpu_metrics_{RUN_STAMP}.csv"
PROMETHEUS_CSV_PATH = f"prometheus_metrics_{RUN_STAMP}.csv"
GPU_SAMPLE_INTERVAL = 1.0  # seconds
PROMETHEUS_SAMPLE_INTERVAL = 1.0  # seconds
LATENCY_CSV_PATH = f"latencies_{RUN_STAMP}.csv"

import requests
port = 30000
url = f"http://localhost:{port}/v1/chat/completions"

prompt = """ ---- ok that was all trash. ignore all of that. now, lets count. when I say count to 10, you say one two three four five six seven eight nine ten. got it? notice how the number of words is also 10. 
when i say count to 100, you say one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty,
twenty-one, twenty-two, twenty-three, twenty-four, twenty-five, twenty-six, twenty-seven, twenty-eight, twenty-nine, thirty,
thirty-one, thirty-two, thirty-three, thirty-four, thirty-five, thirty-six, thirty-seven, thirty-eight, thirty-nine, forty,
forty-one, forty-two, forty-three, forty-four, forty-five, forty-six, forty-seven, forty-eight, forty-nine, fifty,
fifty-one, fifty-two, fifty-three, fifty-four, fifty-five, fifty-six, fifty-seven, fifty-eight, fifty-nine, sixty,
sixty-one, sixty-two, sixty-three, sixty-four, sixty-five, sixty-six, sixty-seven, sixty-eight, sixty-nine, seventy,
seventy-one, seventy-two, seventy-three, seventy-four, seventy-five, seventy-six, seventy-seven, seventy-eight, seventy-nine, eighty,
eighty-one, eighty-two, eighty-three, eighty-four, eighty-five, eighty-six, eighty-seven, eighty-eight, eighty-nine, ninety,
ninety-one, ninety-two, ninety-three, ninety-four, ninety-five, ninety-six, ninety-seven, ninety-eight, ninety-nine, one hundred. Notice how the number of words is 100. ok. Now your turn. count to """

import random
import string

def random_trash_words(n=10, min_len=3, max_len=10):
    """
    Generate n random alphabetic 'trash' words.

    Args:
        n (int): Number of words to generate.
        min_len (int): Minimum word length.
        max_len (int): Maximum word length.

    Returns:
        list[str]: List of random alphabetic words.
    """
    letters = string.ascii_lowercase
    words = [
        ''.join(random.choice(letters) for _ in range(random.randint(min_len, max_len)))
        for _ in range(n)
    ]
    return " ".join(words)


class PrometheusMonitor(threading.Thread):
    """
    Polls Prometheus metrics endpoint and writes CSV rows with metric data
    """
    def __init__(self, csv_path: str, port: int, interval: float = 1.0, stop_event: threading.Event | None = None):
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.port = port
        self.interval = interval
        self.stop_event = stop_event or threading.Event()
        self.metrics_url = f"http://localhost:{port}/metrics"

    def run(self):
        first_write = not os.path.exists(self.csv_path)
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if first_write:
                    writer.writerow(["timestamp", "metric_name", "value"])

                while not self.stop_event.is_set():
                    try:
                        response = requests.get(self.metrics_url)
                        if response.status_code == 200:
                            timestamp = datetime.now().isoformat()
                            metrics_text = response.text
                            # Parse metrics text and write to CSV
                            for line in metrics_text.split('\n'):
                                if line and not line.startswith('#'):
                                    try:
                                        name, value = line.split(' ')
                                        writer.writerow([timestamp, name, value])
                                    except ValueError:
                                        continue
                            f.flush()
                    except requests.RequestException as e:
                        print(f"Error fetching Prometheus metrics: {e}")
                    time.sleep(self.interval)
        except OSError as e:
            print(f"Could not open Prometheus metrics CSV file {self.csv_path}: {e}")

class NvidiaSMIMonitor(threading.Thread):
    """
    Polls `nvidia-smi` and writes CSV rows:
    timestamp,index,name,mem_total_MiB,mem_used_MiB,mem_free_MiB,util_gpu_pct,util_mem_pct
    """
    def __init__(self, csv_path: str, interval: float = 1.0, stop_event: threading.Event | None = None):
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.interval = interval
        self.stop_event = stop_event or threading.Event()

    def run(self):
        header = ["timestamp", "gpu_index", "gpu_name",
                  "mem_total_MiB", "mem_used_MiB", "mem_free_MiB",
                  "util_gpu_pct", "util_mem_pct"]
        first_write = not os.path.exists(self.csv_path)

        # Open once and keep flushing for low overhead
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if first_write:
                    writer.writerow(header)

                cmd = [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ]

                while not self.stop_event.is_set():
                    try:
                        out = subprocess.check_output(cmd, text=True)
                        for line in out.strip().splitlines():
                            # line looks like: "2025/10/07 12:34:56.123, 0, NVIDIA A100, 81251, 1042, 80209, 3, 5"
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) == 8:
                                writer.writerow(parts)
                        f.flush()
                    except FileNotFoundError:
                        print("nvidia-smi not found; GPU monitoring disabled.")
                        break
                    except subprocess.CalledProcessError as e:
                        print(f"nvidia-smi error: {e}")
                    time.sleep(self.interval)
        except OSError as e:
            print(f"Could not open GPU CSV file {self.csv_path}: {e}")


def good_request():
    data = {
        "model": "qwen/qwen2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": random_trash_words(50) + prompt + "50"}],
    }
    start = time.time()
    response = requests.post(url, json=data)
    print(response.json())
    rj = response.json()
    
    return rj.get("usage", {})

def bad_request():
    data = {
        "model": "qwen/qwen2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": random_trash_words(5000) + prompt + "1000"}],
    }

    response = requests.post(url, json=data)
    print(response.json())
    
    rj = response.json()
    
    return rj.get("usage", {})

class TimedThreadPool:
    def __init__(self, max_threads=100, launch_interval=1.0, total_tasks=50):
        self.max_threads = max_threads
        self.launch_interval = launch_interval
        self.total_tasks = total_tasks  # stop after this many tasks (change/remove if you want infinite)
        self.task_queue = queue.Queue()
        self.sem = threading.BoundedSemaphore(value=max_threads)  # concurrency cap
        self.active_threads = []  # optional: track for joins/clean exit
        self.lock = threading.Lock()  # protects shared lists and active_threads

    def regular_task(self, user_id):
        start = time.perf_counter()
        # simulate fast/normal latency
        res = good_request()
        dur = time.perf_counter() - start
        with self.lock:
            regular_latencies.append((start,dur, "good", res.get("prompt_tokens"), res.get("completion_tokens"), res.get("total_tokens")))

    def bad_client_task(self, user_id):
        start = time.perf_counter()
        # simulate slower/problematic latency
        res = bad_request()
        dur = time.perf_counter() - start
        with self.lock:
            bad_client_latencies.append((start,dur, "bad", res.get("prompt_tokens"), res.get("completion_tokens"), res.get("total_tokens")))

    def worker(self, task_id):
        # We already acquired the semaphore before creating the thread.
        try:
            # Decide which task to run
            if IS_GOOD_TASK(task_id):
                self.regular_task(task_id % 10 + 1)
            elif task_id > BAD_TASK_ID_START:
                self.bad_client_task(task_id % 10 + 1)

            print(f"Thread {task_id} started")
            # Simulate some extra work time
            time.sleep(random.uniform(2, 5))
            print(f"Thread {task_id} finished")
        finally:
            # release capacity no matter what
            self.sem.release()
            # remove from active list
            with self.lock:
                try:
                    self.active_threads.remove(threading.current_thread())
                except ValueError:
                    pass  # in case it's not there

    def start(self):
        task_id = 0
        while True:
            # Try to acquire capacity without blocking.
            if self.sem.acquire(blocking=False):
                t = threading.Thread(target=self.worker, args=(task_id,), daemon=True)
                with self.lock:
                    self.active_threads.append(t)
                t.start()
                task_id += 1
            else:
                print("Max threads running, waiting...")

            # Optional exit condition based on total_tasks
            if self.total_tasks is not None and task_id >= self.total_tasks:
                # Wait for all currently running tasks to finish
                while True:
                    with self.lock:
                        alive = [th for th in self.active_threads if th.is_alive()]
                    if not alive:
                        print("All tasks complete, exiting.")
                        return
                    time.sleep(0.1)

            time.sleep(self.launch_interval)

# Example usage
if __name__ == "__main__":
    # Launch a new thread every 1 second, up to 100 concurrent threads, for 50 total tasks.
    stop_event = threading.Event()
    
    # Start GPU monitoring
    gpu_monitor = NvidiaSMIMonitor(csv_path=GPU_CSV_PATH,
                                   interval=GPU_SAMPLE_INTERVAL,
                                   stop_event=stop_event)
    gpu_monitor.start()
    
    # Start Prometheus monitoring
    prom_monitor = PrometheusMonitor(csv_path=PROMETHEUS_CSV_PATH,
                                    port=port,
                                    interval=PROMETHEUS_SAMPLE_INTERVAL,
                                    stop_event=stop_event)
    prom_monitor.start()
    
    pool = TimedThreadPool(max_threads=MAX_THREADS, launch_interval=LAUNCH_INTERVAL, total_tasks=TOTAL_TIME//LAUNCH_INTERVAL)    
    
    try:
        pool.start()
    finally:
        stop_event.set()
        gpu_monitor.join(timeout=5)
        prom_monitor.join(timeout=5)

    # Combine good and bad latencies and dump once
    all_latencies = regular_latencies + bad_client_latencies
    all_latencies.sort(key=lambda x: x[0])  # sort by timestamp

    with open(LATENCY_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_start_iso", "duration_s", "kind", "prompt_tokens", "completion_tokens", "total_tokens"])
        for ts, dur, kind, prompt_tokens, completion_tokens, total_tokens in all_latencies:
            writer.writerow([datetime.fromtimestamp(ts).isoformat(timespec="milliseconds"),
                             f"{dur:.6f}", kind,  prompt_tokens, completion_tokens, total_tokens])

    print(f"Latency CSV saved to: {LATENCY_CSV_PATH}")
    print(f"GPU metrics saved to: {GPU_CSV_PATH}")
    print(f"Regular task samples: {len(regular_latencies)}")
    print(f"Bad client task samples: {len(bad_client_latencies)}")

    # You can inspect the collected latencies after completion:
    print(f"Regular task samples: {len(regular_latencies)}")
    print(f"Bad client task samples: {len(bad_client_latencies)}")
