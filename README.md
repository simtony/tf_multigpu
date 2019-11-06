# Introduction
This repo contains several muti-gpu/distributed training implementation in Tensorflow.

It serve as a good starter training script:

* allreduce_local_sync.py: synchronized update with allreduce. Recommended.
* ps_local_sync.py: synchronized update with parameter server. This is recommended by Tensorflow documentation, though it is the least efficient.
* ps_distribute.py: synchronized/asynchronized update with parameter server. Distributed implementation.

There is a complementary article written in Chinese: https://zhuanlan.zhihu.com/p/50116885

# Usage
The code does not consumes any data: it generates fake data and fits them.
The code shall be ran with Tensorflow >= 1.12. For Tensorflow 2.0, distributed strategy is recommended.
Make sure you have multiple gpus :)

Simply run
```
python allreduce_local_sync.py --gpus 0,1 --max_step 10000
python ps_local_sync.py --gpus 0,1 --max_step 10000
```
For `ps_distribute.py`, make sure you have 2 GPU, and run the following commands in order:
```
# Cluster configuration is specified in the code, in this case, 1 ps and 2 worker
# First start ps:
$ python ps_distribute.py --job ps --index 0

# Then start the workers:
$ python ps_distribute.py --job worker --index 0 --gpu 0 --max_step 10000
$ python ps_distribute.py --job worker --index 1 --gpu 1 --max_step 10000
```

For benchmarking, typically for different communication/computation ratio, simply modify the complexity of the model.


