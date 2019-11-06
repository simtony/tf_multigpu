# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import argparse
import os
import re
import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) >= version.parse("1.13.0"):
    # tf 1.13.0 move nccl from contrib into core
    from tensorflow.python.ops.nccl_ops import all_sum
else:
    from tensorflow.contrib.nccl import all_sum


def split_batches(num_splits, batches):
    batch_size = tf.shape(batches[0])[0]
    # evenly distributed sizes
    divisible_sizes = tf.fill([num_splits], tf.floor_div(batch_size, num_splits))
    remainder_sizes = tf.sequence_mask(tf.mod(batch_size, num_splits),
                                       maxlen=num_splits,
                                       dtype=tf.int32)
    frag_sizes = divisible_sizes + remainder_sizes

    batch_frags_list = []
    for batch in batches:
        batch_frags = tf.split(batch, frag_sizes, axis=0)
        batch_frags_list.append(batch_frags)

    frag_batches_list = list(zip(*batch_frags_list))
    # fix corner case
    for i, frag_batches in enumerate(frag_batches_list):
        if len(frag_batches) == 1:
            frag_batches_list[i] = frag_batches[0]

    return frag_batches_list


def build_dataset(num_gpus):
    rand = tf.random_normal([10000, 300])
    rand_labels = tf.random_uniform([10000], minval=0, maxval=9, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((rand, rand_labels))
    dataset = dataset.repeat()
    dataset = dataset.batch(100)
    dataset = dataset.map(lambda rand, rand_labels: split_batches(num_splits=num_gpus, batches=[rand, rand_labels]))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0,1', type=str)
    parser.add_argument('--max_step', default=10000, type=int)
    args = parser.parse_args()
    args.num_gpus = len(args.gpus.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # avoid unimplemented gpu kernel error
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        dataset = build_dataset(args.num_gpus)
        iterator = dataset.make_initializable_iterator()
        tower_batches = iterator.get_next()

        tower_grads_list = []
        tower_tvars_list = []
        tower_gvars_list = []
        tower_loss_list = []
        for index, tower_batch in enumerate(tower_batches):
            # by-device variable scope
            with tf.variable_scope("tower_%d" % index) as scope, \
                    tf.device('/gpu:%d' % index):

                tower_loss = build_tower(tower_batch)
                tower_gvars = tf.global_variables(scope._name)
                tower_tvars = tf.trainable_variables(scope._name)
                tower_grads = tf.gradients(tower_loss, tower_tvars)

                tower_loss_list.append(tower_loss)
                tower_tvars_list.append(tower_tvars)
                tower_gvars_list.append(tower_gvars)
                tower_grads_list.append(tower_grads)

                if index == 0:
                    # only one variable global saver
                    def clean(name):
                        name = re.sub('^tower_\d+/', '', name)
                        name = re.sub(':\d+$', '', name)
                        return name

                    save_dict = {clean(var.name): var
                                 for var in tower_gvars}
                    saver = tf.train.Saver(save_dict)

        with tf.name_scope("tower_gvar_sync"):
            # different device is init with different random seed
            # need explicit synchronization before training!!!
            if len(tower_gvars_list) == 1:
                tower_gvar_sync = tf.no_op()
            else:
                sync_ops = []
                for vars in zip(*tower_gvars_list):
                    for var in vars[1:]:
                        sync_ops.append(tf.assign(var, vars[0]))
                tower_gvar_sync = tf.group(*sync_ops)

        with tf.name_scope('all_reduce'):
            avg_tower_grads_list = []
            for grads_to_avg in zip(*tower_grads_list):
                # nccl.all_sum will automatically
                # convert sparse gradients into dense one
                avg_tower_grads_list.append(all_sum(grads_to_avg))
            avg_tower_grads_list = zip(*avg_tower_grads_list)

        with tf.name_scope('metrics'):
            loss = tf.add_n(tower_loss_list) / len(tower_loss_list)

        train_ops = []
        for index, (tower_vars, tower_grads) in \
                enumerate(zip(tower_tvars_list, avg_tower_grads_list)):
            with tf.variable_scope("tower_%d" % index), \
                 tf.device('/gpu:%d' % index):
                tower_grads = [grad / len(tower_batches) for grad in tower_grads]
                if index == 0:
                    # only increment global step with the first worker
                    step = tf.train.get_or_create_global_step()

                tower_optimizer = tf.train.AdamOptimizer()
                tower_train_op = tower_optimizer.apply_gradients(zip(tower_grads, tower_vars),
                                                                 global_step=step if index == 0 else None)
                train_ops.append(tower_train_op)
        train_op = tf.group(train_ops)

        # start running
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        # important to sync variables before training!
        sess.run(tower_gvar_sync)
        while True:
            try:
                fetch_loss, fetch_step, _ = sess.run([loss, step, train_op])
                if fetch_step % 20 == 0:
                    print("step: %d, loss: %.4f" % (fetch_step, fetch_loss))
                if fetch_step > args.max_step:
                    break
            except tf.errors.OutOfRangeError:
                break
        saver.save(sess, "./model")


if __name__ == '__main__':
    main()
