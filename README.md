# Distributed Tensorflow Weirdness
This includes a couple scripts to test the sharding solution given over slack. You can
just run python ps_train.py to train a toy model with keras using 1 ps and 2 workers locally.

The only libraries needed are tensorflow and portpicker. `pip install tensorflow portpicker` should be sufficient. Python 3.9 was used to produce this although 3.7/3.8 probably work.
