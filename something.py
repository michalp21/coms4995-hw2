from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import scipy.misc
import glob

import tensorflow as tf

LEARNING_RATE = .001
MAX_ITER = 1000

# parameters = {}
# num_layers = len(layer_dimensions)
# drop_prob = drop_prob
# reg_lambda = reg_lambda

def train(X, y, iters=1000, alpha=0.0001, batch_size=100, print_every=100):
    """
    :param X: input samples, each column is a sample
    :param y: labels for input samples, y.shape[0] must equal X.shape[1]
    :param iters: number of training iterations
    :param alpha: step size for gradient descent
    :param batch_size: number of samples in a minibatch
    """

    # Input placeholders, None indicates variable size (for batches)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    sess = tf.InteractiveSession()

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    # get minibatch
    X_batch, y_batch = get_batch(X, y, batch_size)

    # input
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    # dense layer
    hidden_1 = tf.layers.Dense(inputs=input_layer,
                                    units=500,
                                    activation=tf.nn.relu)
    variable_summaries(hidden_1.kernel)
    
    # predictions
    y_hat = tf.layers.Dense(inputs=hidden_1,
                                units=10,
                                activation=tf.identity)
    variable_summaries(y_hat.kernel)

    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_hat)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to cifar10-hw2 (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('cifar10-hw2/train', sess.graph)
    test_writer = tf.summary.FileWriter('cifar10-hw2/test')
    tf.global_variables_initializer().run()

    def feed_dict():
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(MAX_ITER):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict())
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()

def loadData:
    # Load the data
    data_root_path = 'cifar10-hw2/'
    X_train, y_train = get_train_data(data_root_path + 'train') # this may take a few minutes
    X_test = get_images(data_root_path + 'test')
    #X_test, y_test = get_train_data(base_path, 'test')
    print(X_train.shape)
    print(y_train.shape)
    print('Data loading done')

train()