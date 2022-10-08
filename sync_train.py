from utils import create_tf_configs, run_tasks

tf_configs = create_tf_configs(2, 0)
run_tasks("sync", tf_configs, "sync_logs")
