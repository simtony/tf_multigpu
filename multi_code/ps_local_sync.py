# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import argparse
import os
import tensorflow as tf


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

    dataset = build_dataset(args.num_gpus)
    iterator = dataset.make_initializable_iterator()
    tower_batches = iterator.get_next()

    # build train graph
    tower_grads_list = []
    tower_loss_list = []
    # global variable scope
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        for index, tower_batch in enumerate(tower_batches):
            with tf.device('/gpu:%d' % index):
                tower_loss = build_tower(tower_batch)
                if index == 0:
                    # variables are first created on gpu 0
                    # so gpu 0 is the parameter server
                    tvars = tf.trainable_variables()
                tower_grads = tf.gradients(tower_loss, tvars)
                tower_grads_list.append(tower_grads)
                tower_loss_list.append(tower_loss)

    with tf.device('/gpu:0'):
        # update on parameter server
        loss = tf.add_n(tower_loss_list) / args.num_gpus
        avg_grads = []
        for grad_list in zip(*tower_grads_list):
            # avoid making sparse gradients dense with simple tf.add_n
            grad_avg = tf.reduce_mean(tf.stack(grad_list, axis=0), axis=0)
            avg_grads.append(grad_avg)

        step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.apply_gradients(zip(avg_grads, tvars),
                                             global_step=step)
    saver = tf.train.Saver()

    # start running
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
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
