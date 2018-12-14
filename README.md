# Introduction
This repo contains several muti-gpu/distributed training implementation in Tensorflow.

It serve as benchmarking as well as a good starter training script:

* allreduce_local_sync.py: synchronized update with allreduce. Recommended.
* ps_local_sync.py: synchronized update with parameter server. This is recommended by Tensorflow documentation, but it is the least efficient.
* ps_distribute.py: synchronized/asynchronized update with parameter server. Distributed implementation.
There is a complementary article though written in Chinese: https://zhuanlan.zhihu.com/p/50116885
# How to use
The code does not consumes any data: it generate fake data and fit them.

Simply run
``python <filename>``

For benchmarking,especially for different communication/computation ratio, simply modify the fake model.



