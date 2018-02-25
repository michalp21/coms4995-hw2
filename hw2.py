import tensorflow as tf
import numpy as np

from hw1_helpers import *

data_root_path = '/home/daniel/cifar10-hw2/'

BATCH_SIZE = 100

X_test = get_images(data_root_path + 'test')
X_train, y_train = get_train_data(data_root_path)

X_train = X_train.T

print('Data loading hw1 done: %s')

# enqueue_many=True because we loaded all the images and it's more like we need
# to queue them all to batch them, then accumulate them and dbatch them.

batch = tf.random_uniform([BATCH_SIZE], minval=0, maxval=50000, dtype=tf.int32)

X_batch = tf.gather(X_train, batch)
y_batch = tf.gather(y_train, batch)


# Probably need to reshape to use convolutions to 32x32x3
#X_batch = tf.reshape(X_batch, [-1, 3str(2), 32, 3])
dense = tf.layers.dense(inputs=X_batch, units=500, activation=tf.nn.relu)

logits = tf.layers.dense(inputs=dense, units=10)

onehot = tf.one_hot(indices=y_batch, depth=10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot, logits=logits)

# train
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train_op = optimizer.minimize(
	loss=loss,
	global_step=tf.train.get_global_step())

print("Try training")

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess)

for i in range(1000):
    _, loss_value = sess.run((train_op, loss))
    print(str(i) + ": " + str(loss_value))
print("Done training")


# eval

# predict