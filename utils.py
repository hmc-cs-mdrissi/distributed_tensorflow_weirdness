import json
import os
import subprocess
import sys
from typing import Any, Dict, Sequence

from portpicker import pick_unused_port

from toy_train import main


def create_tf_configs(worker_count: int, ps_count: int):
    cluster_dict = {}
    if worker_count:
        cluster_dict["worker"] = [f"localhost:{pick_unused_port()}" for _ in range(worker_count)]
    if ps_count:
        cluster_dict["ps"] = [f"localhost:{pick_unused_port()}" for _ in range(ps_count)]

    cluster_dict["chief"] = [f"localhost:{pick_unused_port()}"]

    tf_configs = []
    for i in range(worker_count):
        tf_configs.append({"cluster": cluster_dict, "task": {"type": "worker", "index": i}})

    for i in range(ps_count):
        tf_configs.append({"cluster": cluster_dict, "task": {"type": "ps", "index": i}})

    tf_configs.append({"cluster": cluster_dict, "task": {"type": "chief", "index": 0}})

    return tf_configs

def run_tasks(distributed_type: str, tf_configs: Sequence[Dict[str, Any]], log_dir: str):
    command = [sys.executable, "toy_train.py", "--distributed-type", distributed_type]
    for tf_config in tf_configs[:-1]:
        env = os.environ.copy()
        env["TF_CONFIG"] = json.dumps(tf_config)

        name = tf_config["task"]["type"] + "_" + str(tf_config["task"]["index"])
        os.makedirs(log_dir, exist_ok=True)
        log_file_stdout = os.path.join(log_dir, f"stdout_{name}.log")
        log_file_stderr = os.path.join(log_dir, f"stderr_{name}.log")

        with open(log_file_stdout, "w") as stdout, open(log_file_stderr, "w") as stderr:
            subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)

    chief_config = tf_configs[-1]
    os.environ["TF_CONFIG"] = json.dumps(chief_config)
    main(distributed_type)
