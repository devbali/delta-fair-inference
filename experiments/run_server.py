
import os
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

# This is equivalent to running the following command in your terminal
# python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0


host = os.environ.get("RUN_SERVER_HOST", "127.0.0.1")
port = int(os.environ.get("RUN_SERVER_PORT", "30000"))

server_process, port = launch_server_cmd(
    f"""python3.12 -m sglang.launch_server --model-path openai/gpt-oss-20b \
 --host {host} --device cuda --log-level warning
""",
    host=host,
    port=port,
)

wait_for_server(f"http://localhost:{port}")
print(f"Server is running at http://localhost:{port}")
