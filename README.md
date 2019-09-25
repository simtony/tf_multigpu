# Introduction
This repo contains several muti-gpu/distributed training implementation in Tensorflow.

It serve as a good starter training script:

* allreduce_local_sync.py: synchronized update with allreduce. Recommended.
* ps_local_sync.py: synchronized update with parameter server. This is recommended by Tensorflow documentation, though it is the least efficient.
* ps_distribute.py: synchronized/asynchronized update with parameter server. Distributed implementation.

There is a complementary article, although written in Chinese: https://zhuanlan.zhihu.com/p/50116885

# Usage
The code does not consumes any data: it generates fake data and fits them.
The code shall be ran with tensorflow-1.12 with multiple GPUs.

Simply run
```
python allreduce_local_sync.py
python ps_local_sync.py
```
For `ps_distribute.py`, make sure you have 2 GPU, and run the following commands in order:
```
python async.py --job ps --index 0
python async.py --job woker --index 0
python async.py --job woker --index 1
```

For benchmarking, especially for different communication/computation ratio, simply modify the model part.
