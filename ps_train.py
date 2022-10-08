from utils import create_tf_configs, run_tasks

tf_configs = create_tf_configs(2, 1)
run_tasks("ps", tf_configs, "ps_logs")
