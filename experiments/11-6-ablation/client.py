import argparse
import threading
import time
import random
import math
from datetime import datetime
import os
import csv
import subprocess
import asyncio
import aiohttp
import json

latencies = []
DEFAULT_MAX_THREADS = 1000  # max concurrent threads
DEFAULT_REQUEST_RATE = 5.0  # requests per second
DEFAULT_DURATION = 300.0  # total run time in seconds
DEFAULT_PROMPT_WORDS = 50
DEFAULT_COMPLETION_TOKENS = 50

GET_USER = lambda task_id: task_id % 20 + 1

USER_ID_PREFILL_TOKENS = {i: 0 for i in range(1, 21)}
USER_ID_DECODE_TOKENS = {i: 0 for i in range(1, 21)}
PREFILL_LAST_CHECKED_TIMESTAMP = 0
DECODE_LAST_CHECKED_TIMESTAMP = 0

RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
GPU_CSV_PATH = f"gpu_metrics_{RUN_STAMP}.csv"
PROMETHEUS_CSV_PATH = f"prometheus_metrics_{RUN_STAMP}.csv"
GPU_SAMPLE_INTERVAL = 1.0  # seconds
PROMETHEUS_SAMPLE_INTERVAL = 1.0  # seconds
TOKEN_USAGE_CSV_PATH = f"token_usage_{RUN_STAMP}.csv"
TOKEN_SAMPLE_INTERVAL = 1.0  # seconds
LATENCY_CSV_PATH = f"latencies_{RUN_STAMP}.csv"

import requests
port = 30000
url = f"http://localhost:{port}/generate"

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


class LatencyMonitor(threading.Thread):
    """
    Periodically writes latency data from regular and bad client tasks to CSV file
    """
    def __init__(self, csv_path: str, interval: float = 1.0, stop_event: threading.Event | None = None):
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.interval = interval
        self.stop_event = stop_event or threading.Event()
        self.last_write_index = 0

    def run(self):
        first_write = not os.path.exists(self.csv_path)
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if first_write:
                    writer.writerow(["timestamp_start_iso", "user", "kind", "duration_ttft", "duration_taft", "prompt_tokens", "completion_tokens"])

                while not self.stop_event.is_set():
                    # Combine and sort new latencies
                    self.last_write_index, latest = len(latencies), latencies[self.last_write_index:]
                    latest.sort(key=lambda x: x[0])

                    # Write any new entries
                    for ts, user, kind, duration_ttft, duration_taft, prompt_tokens, completion_tokens in latest:
                        writer.writerow([datetime.fromtimestamp(ts).isoformat(timespec="milliseconds"),
                                      user, kind, f"{duration_ttft:.6f}", f"{duration_taft:.6f}", prompt_tokens, completion_tokens])
                    f.flush()
                    time.sleep(self.interval)
        except OSError as e:
            print(f"Could not open latency CSV file {self.csv_path}: {e}")

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

class TokenUsageMonitor(threading.Thread):
    """
    Periodically snapshots USER_ID_PREFILL_TOKENS/USER_ID_DECODE_TOKENS into a CSV.
    """
    def __init__(self, csv_path: str, interval: float = 1.0, stop_event: threading.Event | None = None):
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.interval = interval
        self.stop_event = stop_event or threading.Event()

    def run(self):
        header = ["timestamp_iso", "user_id", "prefill_tokens", "decode_tokens"]
        first_write = not os.path.exists(self.csv_path)
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if first_write:
                    writer.writerow(header)

                while not self.stop_event.is_set():
                    timestamp = datetime.now().isoformat(timespec="seconds")
                    prefill_snapshot = USER_ID_PREFILL_TOKENS.copy()
                    decode_snapshot = USER_ID_DECODE_TOKENS.copy()
                    user_ids = sorted(set(prefill_snapshot) | set(decode_snapshot))

                    for user_id in user_ids:
                        writer.writerow([
                            timestamp,
                            user_id,
                            prefill_snapshot.get(user_id, 0),
                            decode_snapshot.get(user_id, 0),
                        ])
                    f.flush()
                    time.sleep(self.interval)
        except OSError as e:
            print(f"Could not open token usage CSV file {self.csv_path}: {e}")

class TimedAsyncPool:
    def __init__(self, max_concurrent, launch_interval=1.0, total_tasks=50,
                 prompt_word_count=DEFAULT_PROMPT_WORDS,
                 completion_tokens=DEFAULT_COMPLETION_TOKENS):
        self.max_concurrent = max_concurrent
        self.launch_interval = launch_interval
        self.total_tasks = total_tasks
        self.sem = asyncio.Semaphore(value=max_concurrent)
        self.session = None
        self.prompt_word_count = prompt_word_count
        self.completion_tokens = completion_tokens
        
    async def setup(self):
        # Create aiohttp session for all requests
        connector = aiohttp.TCPConnector(limit=None)
        self.session = aiohttp.ClientSession(connector=connector)
        
    async def cleanup(self):
        if self.session:
            await self.session.close()
    
    async def do_request (self, user_id, data,sampling_params, kind):
        start = time.time()
        sampling = {
                **sampling_params
            }
        data = {
            "user": str(user_id),
            "uid": f"user_{user_id}",
            "stream": True,
            "sampling_params": sampling_params,
            **data
        }
        
        ttft = None
        prefill_token_count = 0
        decode_token_count  = 0
        
        async with self.session.post(url, json=data, timeout=None) as response:
            async for chunk, end_of_chunk in response.content.iter_chunks():
                if ttft is None:
                    ttft = time.time() - start
                chunk = chunk.decode("utf-8")
                if chunk and chunk.startswith("data:"):
                    try:
                        data = json.loads(chunk[5:].strip("\n"))
                        if "text" not in data:
                            print(data, chunk)
                            raise ValueError("No text field in response chunk")

                        output = data["text"]
                        #print("output chunk:", data)

                        USER_ID_PREFILL_TOKENS[user_id] += data['meta_info']['prompt_tokens'] - prefill_token_count
                        USER_ID_DECODE_TOKENS[user_id] += data['meta_info']['completion_tokens'] - decode_token_count
                        prefill_token_count = data['meta_info']['prompt_tokens']
                        decode_token_count = data['meta_info']['completion_tokens']

                        if data["meta_info"]["finish_reason"] is not None:
                            prompt_tokens = data["meta_info"].get("prompt_tokens")
                            completion_tokens = data["meta_info"].get("completion_tokens")
                    
                    except json.JSONDecodeError as e:
                        continue

        if ttft is None:
            raise ValueError("No content received in response stream")

        taft = time.time() - start - ttft

        latencies.append((start,user_id, kind, ttft, taft,
                         prompt_tokens,
                         completion_tokens,
                        ))
        print(f"User {user_id} {kind} request completed: TTFT {ttft:.3f}s, TAFT {taft:.3f}s, prompt tokens {prompt_tokens}, completion tokens {completion_tokens}. Output: {output[:60]}...")

    async def fixed_request(self, user_id):
        prompt_text = random_trash_words(self.prompt_word_count) + prompt + str(self.completion_tokens)
        sampling = {
            "min_new_tokens": self.completion_tokens,
            "max_new_tokens": self.completion_tokens,
        }
        return await self.do_request(
            user_id,
            {"text": prompt_text},
            sampling,
            "fixed",
        )

    async def worker(self, task_id):
        async with self.sem:  # Use semaphore as async context manager
            print(f"Task {task_id} started sem value {self.sem._value}")
            await self.fixed_request(GET_USER(task_id))

            print(f"Task {task_id} finished")

    async def start(self):
        await self.setup()
        try:
            task_id = 0
            tasks = set()
            
            while True:
                if task_id >= self.total_tasks:
                    # Wait for remaining tasks
                    if tasks:
                        await asyncio.gather(*tasks)
                    print("All tasks complete, exiting.")
                    break
                
                # Create new task
                task = asyncio.create_task(self.worker(task_id))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                task_id += 1
                
                await asyncio.sleep(self.launch_interval)
        finally:
            await self.cleanup()

# Example usage
def parse_args():
    parser = argparse.ArgumentParser(description="Run the 11-6 ablation client with fixed token budgets.")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_THREADS,
                        help="Maximum number of concurrent requests (default: %(default)s)")
    parser.add_argument("--request-rate", type=float, default=DEFAULT_REQUEST_RATE,
                        help="Requests launched per second (default: %(default)s)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help="Total run time in seconds (default: %(default)s)")
    parser.add_argument("--prompt-words", type=int, default=DEFAULT_PROMPT_WORDS,
                        help="Number of random trash words prepended before the prompt (default: %(default)s)")
    parser.add_argument("--completion-tokens", type=int, default=DEFAULT_COMPLETION_TOKENS,
                        help="Tokens requested via len_output_tokens/max_new_tokens (default: %(default)s)")

    args = parser.parse_args()
    if args.request_rate <= 0:
        parser.error("--request-rate must be greater than zero")
    if args.duration <= 0:
        parser.error("--duration must be greater than zero")
    if args.prompt_words <= 0:
        parser.error("--prompt-words must be greater than zero")
    if args.completion_tokens <= 0:
        parser.error("--completion-tokens must be greater than zero")
    if args.max_concurrent <= 0:
        parser.error("--max-concurrent must be greater than zero")
    return args

DO_PROMETHEUS = False
if __name__ == "__main__":
    args = parse_args()
    launch_interval = 1.0 / args.request_rate
    total_tasks = max(1, math.ceil(args.request_rate * args.duration))
    print(f"Configured run: {total_tasks} requests over {args.duration}s at {args.request_rate} req/s "
          f"({args.max_concurrent} max concurrent, {args.prompt_words} prompt words, {args.completion_tokens} completion tokens)")
    stop_event = threading.Event()
    
    # Start GPU monitoring
    gpu_monitor = NvidiaSMIMonitor(csv_path=GPU_CSV_PATH,
                                   interval=GPU_SAMPLE_INTERVAL,
                                   stop_event=stop_event)
    gpu_monitor.start()

    # Start token usage monitoring
    token_monitor = TokenUsageMonitor(csv_path=TOKEN_USAGE_CSV_PATH,
                                      interval=TOKEN_SAMPLE_INTERVAL,
                                      stop_event=stop_event)
    token_monitor.start()
    
    # Start Prometheus monitoring
    prom_monitor = PrometheusMonitor(csv_path=PROMETHEUS_CSV_PATH,
                                    port=port,
                                    interval=PROMETHEUS_SAMPLE_INTERVAL,
                                    stop_event=stop_event)

    
    # Start latency monitoring
    latency_monitor = LatencyMonitor(csv_path=LATENCY_CSV_PATH,
                                   interval=1.0,
                                   stop_event=stop_event)
    
    latency_monitor.start()
    if DO_PROMETHEUS:
        prom_monitor.start()
    
    pool = TimedAsyncPool(
        max_concurrent=args.max_concurrent,
        launch_interval=launch_interval,
        total_tasks=total_tasks,
        prompt_word_count=args.prompt_words,
        completion_tokens=args.completion_tokens,
    )    
    
    try:
        # Run the async pool in the event loop
        asyncio.run(pool.start())
    finally:
        stop_event.set()
        gpu_monitor.join(timeout=5)
        token_monitor.join(timeout=5)
        if DO_PROMETHEUS:
            prom_monitor.join(timeout=5)
        latency_monitor.join(timeout=5)

    print(f"GPU metrics saved to: {GPU_CSV_PATH}")
    print(f"Token usage metrics saved to: {TOKEN_USAGE_CSV_PATH}")
    print(f"Task samples: {len(latencies)}")
