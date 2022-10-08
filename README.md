# Distributed Tensorflow Weirdness
This includes a couple scripts to reproduce difference in behavior for ParameterServer vs MultiWorkerMirrored training. The scripts train a toy model on toy data and print logs of which data points are processed by each worker. In parameter server each worker recieves entire dataset, but in synchronous training the dataset is automatically sharded.

The log files are also included in the repository. To generate logs just run `python ps_train.py` or `python sync_train.py`.
