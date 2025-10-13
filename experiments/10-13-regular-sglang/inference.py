from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

# This is equivalent to running the following command in your terminal
#  python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct  --host localhost --log-level info --enable-metrics

server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct  --enable-metrics --host localhost --log-level info --disable-cuda-graph --disable-flashinfer  --mem-fraction-static 0.8
""")

wait_for_server(f"http://localhost:{port}")

