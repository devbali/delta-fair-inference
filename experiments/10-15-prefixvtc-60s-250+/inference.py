from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

# This is equivalent to running the following command in your terminal
#  python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct  --host localhost --log-level info --enable-metrics

server_process, port = launch_server_cmd(
    """
export HF_TOKEN= ;
python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --host localhost --log-level info --schedule-policy lpm_mdrr --Q 60000  --dtype float16
"""
)

wait_for_server(f"http://localhost:{port}")

