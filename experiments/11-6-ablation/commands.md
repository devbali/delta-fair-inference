# Server command for all
```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B --enable-eviction --enable-iterative-eviction --port 30000  --mem-frac 0.7 --schedule-po lpm_mdrr  --load-ba hdoubleq  --Q 50000 --global-Q 100000 --max-total 20000 --max-prefill 20000   --dtype float16
```

# Ablation Study #1: # of tokens relation to prefill and decode time
Run the standard server command above and sweep prompt tokens while keeping completion tokens high (2k).

```bash
# 50 prompt words → 2k completion tokens
python experiments/11-6-ablation/client.py --prompt-words 50 --completion-tokens 2000 --request-rate 1.5 --duration 180 --max-concurrent 300

# 200 prompt words → 2k completion tokens
python experiments/11-6-ablation/client.py --prompt-words 200 --completion-tokens 2000 --request-rate 1.5 --duration 180 --max-concurrent 300

# 500 prompt words → 2k completion tokens
python experiments/11-6-ablation/client.py --prompt-words 500 --completion-tokens 2000 --request-rate 1.5 --duration 180 --max-concurrent 300

# 800 prompt words → 2k completion tokens
python experiments/11-6-ablation/client.py --prompt-words 800 --completion-tokens 2000 --request-rate 1.5 --duration 180 --max-concurrent 300
```

# Ablation Study #2: Batch size and decode time
Stay on the standard server command and vary throughput by changing request rate while keeping prompts tiny and completions large (1.8k tokens).

```bash
# Low request rate baseline
python experiments/11-6-ablation/client.py --prompt-words 10 --completion-tokens 1800 --request-rate 2 --duration 180 --max-concurrent 400

# Medium request rate
python experiments/11-6-ablation/client.py --prompt-words 10 --completion-tokens 1800 --request-rate 6 --duration 180 --max-concurrent 600

# High request rate
python experiments/11-6-ablation/client.py --prompt-words 10 --completion-tokens 1800 --request-rate 12 --duration 180 --max-concurrent 900
```

# Ablation Study #3: Compute share to prefill and decode time
Use the dedicated launcher (`experiments/11-6-ablation/server.py`) to bind the server to a static NVIDIA partition and shrink its CUDA MPS share. Replace `${MIG_PARTITION}` with the MIG slice or GPU index exposed by your NVIDIA container runtime (for example `MIG-GPU-XXXXXXXXXXXXXXXX/1/0`). The client command stays constant.

```bash
# Full compute share (reference)
python experiments/11-6-ablation/server.py --visible-device ${MIG_PARTITION} --compute-share 100 --port 30000
python experiments/11-6-ablation/client.py --prompt-words 600 --completion-tokens 2500 --request-rate 2 --duration 240 --max-concurrent 400

# 60% compute share
python experiments/11-6-ablation/server.py --visible-device ${MIG_PARTITION} --compute-share 60 --port 30000
python experiments/11-6-ablation/client.py --prompt-words 600 --completion-tokens 2500 --request-rate 2 --duration 240 --max-concurrent 400

# 30% compute share
python experiments/11-6-ablation/server.py --visible-device ${MIG_PARTITION} --compute-share 30 --port 30000
python experiments/11-6-ablation/client.py --prompt-words 600 --completion-tokens 2500 --request-rate 2 --duration 240 --max-concurrent 400
```

