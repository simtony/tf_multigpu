# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import argparse
import os

import tensorflow as tf


def build_dataset():
    rand = tf.random_normal([10000, 300])
    rand_labels = tf.random_uniform([10000], minval=0, maxval=9, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((rand, rand_labels))
    dataset = dataset.repeat()
    dataset = dataset.batch(100)
    return dataset


def build_tower(batch):
    feature, label = batch
    matrix = tf.get_variable('matrix', shape=[300, 500])
    middle_matrix = tf.get_variable('middle_matrix', shape=[500, 500])
    out_matrix = tf.get_variable('out_matrix', shape=[500, 10])
    feature = tf.matmul(feature, matrix)
    for i in range(10):
        feature = tf.matmul(feature, middle_matrix)
    logits = tf.matmul(feature, out_matrix)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=label))
    return loss


def main():
    usage = """
    First start ps: 
    $ python ps_distribute.py --job ps --index 0
    
    Then start the workers: 
    $ python ps_distribute.py --job worker --index 0 --gpu 0 --max_step 10000
    $ python ps_distribute.py --job worker --index 1 --gpu 1 --max_step 10000
    """

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--job', choices=['worker', 'ps'])
    parser.add_argument('--index', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--sync', action="store_true", help="Turn on synchronized gradient mode")
    parser.add_argument('--max_step', default=10000, type=int)

    # one ps, two worker
    ps = ["127.0.0.1:60000"]
    worker = ["127.0.0.1:60001",
              "127.0.0.1:60002"]
    args = parser.parse_args()

    cluster = tf.train.ClusterSpec(
            {"ps": ps,
             "worker": worker}
    )

    if args.job == "ps":
        # ps on cpu
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        server = tf.train.Server(cluster,
                                 job_name=args.job,
                                 task_index=args.index)
        server.join()

    elif args.job == "worker":
        # worker on gpu
        # gpu index by worker index
        os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % args.gpu
        server = tf.train.Server(cluster,
                                 job_name=args.job,
                                 task_index=args.index)

        dataset = build_dataset()
        iterator = dataset.make_initializable_iterator()
        batch = iterator.get_next()

        with tf.device(
                tf.train.replica_device_setter(
                        worker_device="/job:worker/task:%d" % args.index,
                        cluster=cluster)):
            loss = build_tower(batch)
            tvars = tf.trainable_variables()
            grads = tf.gradients(loss, tvars)
            step = tf.train.get_or_create_global_step()
            # avoid concurrent update
            optimizer = tf.train.GradientDescentOptimizer(0.1, use_locking=True)

            if args.sync:
                optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=len(worker))
                hooks = [optimizer.make_session_run_hook(args.index == 0)]
            else:
                hooks = []

            train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                 global_step=step)

        # if throws uninitialized errors, put your initializers here
        local_init_op = tf.group(tf.local_variables_initializer(),
                                 tf.tables_initializer(),
                                 iterator.initializer)
        scaffold = tf.train.Scaffold(local_init_op=local_init_op)
        hooks.append(tf.train.StopAtStepHook(args.max_step))

        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=(args.index == 0),
                scaffold=scaffold,
                checkpoint_dir='./',
                save_summaries_steps=15,
                save_checkpoint_steps=10000,
                hooks=hooks) as sess:

            while not sess.should_stop():
                fetch_loss, fetch_step, _ = sess.run([loss,
                                                      step,
                                                      train_op])

                if fetch_step % 20 == 0:
                    print("job: %s, task_index: %d, step: %d, loss: %.4f" %
                          (args.job, args.index, fetch_step, fetch_loss))


if __name__ == '__main__':
    main()
