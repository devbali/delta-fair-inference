# LPM (re run)
## Server
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B --enable-eviction --enable-iterative-eviction --port 30000  --mem-frac 0.7 --schedule-po lpm --Q 4000 --global-Q 6000 --max-total 20000 --max-prefill 20000   --dtype float16

## Client (dir ~/fairinf/delta-fair-inference/Prefix-VTC/benchmark/llm_judge)
python bench_online_llmasjudge.py --port 30000 --result-file Global --num-questions 150 --result_file 1128test_new.jsonl --intervals 2 2 2  --dp 1 --enable-eviction --enable-iterative-eviction  --branches 16 2 2 --initial-delay 10 >$HOME/fairinf/delta-fair-inference/experiments/verify-prefix-vtc/lpm2.txt

## Results
Fastest completion by User user0 in 325.173 seconds
len new_results_from_backend: 1275
Service throughput (token/s): 10232.120
Total service throughput when all 3 have (token/s): 11644.711

Throughput Analysis:
User throughput (req/s, input tok/s, output tok/s): {'user0': [2.8428787592476294, 10591.395473181932, 454.75272438384565], 'user1': [0.025382846064710977, 76.60225656753964, 0.025382846064710977], 'user2': [0.022209990306622105, 67.11224499509582, 0.022209990306622105]}
Minimum completion duration: 325.173s
Total execution duration: 487.173s
Service fairness (Jain index): 0.342
Request fairness (Jain index): 0.345

Latency Statistics per User:
User ID | Mean (s) | P50 (s) | P99 (s)
----------------------------------------
User user0 |  129.269 | 132.727 | 226.949 | 218.599
User user1 |  311.614 | 312.505 | 349.220 | 339.597
User user2 |  396.083 | 401.044 | 429.646 | 425.340


# VTC
## Server
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B --enable-eviction --enable-iterative-eviction --port 30000 --mem-frac 0.7   --schedule-po vtc  --load-ba drr1 --Q 3000 --global-Q 6000 --max-total 20000 --max-prefill 20000   --dtype float16

## Client
python bench_online_llmasjudge.py --port 30000 --result-file Global --num-questions 150 --result_file 1128test_new.jsonl --intervals 2 2 2  --dp 1 --enable-eviction --enable-iterative-eviction  --branches 16 2 2 --initial-delay 10 >$HOME/fairinf/delta-fair-inference/experiments/verify-prefix-vtc/vtc.txt

## Results
Fastest completion by User user1 in 313.575 seconds
len new_results_from_backend: 1289
Service throughput (token/s): 9011.108
Total service throughput when all 3 have (token/s): 8478.732

Throughput Analysis:
User throughput (req/s, input tok/s, output tok/s): {'user0': [1.0409289562448236, 3902.6864188060276, 178.54237454881545], 'user1': [0.5863460576315779, 1991.300387375323, 65.27876305665718], 'user2': [0.5797578996806613, 1969.147706265366, 63.977601861351154]}
Minimum completion duration: 313.575s
Total execution duration: 559.302s
Service fairness (Jain index): 0.886
Request fairness (Jain index): 0.921

Latency Statistics per User:
User ID | Mean (s) | P50 (s) | P99 (s)
----------------------------------------
User user0 |  337.474 | 371.692 | 491.500 | 486.202
User user1 |  122.469 | 136.160 | 213.960 | 187.685
User user2 |  120.563 | 131.757 | 246.360 | 190.465

# LPM MDRR
## Server
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B --enable-eviction --enable-iterative-eviction --port 30000  --mem-frac 0.7 --schedule-po lpm_mdrr  --load-ba hdoubleq  --Q 50000 --global-Q 100000 --max-total 20000 --max-prefill 20000   --dtype float16

## Client
python bench_online_llmasjudge.py --port 30000 --result-file Global --num-questions 150 --result_file 1128test_new.jsonl --intervals 2 2 2  --dp 1 --enable-eviction --enable-iterative-eviction  --branches 16 2 2 --initial-delay 10 >$HOME/fairinf/delta-fair-inference/experiments/verify-prefix-vtc/lpn_mdrr.txt

## Results
Fastest completion by User user2 in 348.400 seconds
len new_results_from_backend: 1285
Service throughput (token/s): 8994.684
Total service throughput when all 3 have (token/s): 8082.561

Throughput Analysis:
User throughput (req/s, input tok/s, output tok/s): {'user0': [1.040190490325579, 3856.0629798890473, 168.1828448178968], 'user1': [0.5230503317830326, 1770.9627259053611, 61.66674759123347], 'user2': [0.5496461113652207, 1863.637197402806, 66.09937752159816]}
Minimum completion duration: 348.400s
Total execution duration: 557.391s
Service fairness (Jain index): 0.866
Request fairness (Jain index): 0.898

Latency Statistics per User:
User ID | Mean (s) | P50 (s) | P99 (s)
----------------------------------------
User user0 |  311.395 | 379.154 | 457.421 | 446.874
User user1 |  142.381 | 126.205 | 241.912 | 220.653
User user2 |  158.145 | 138.696 | 252.088 | 243.658

