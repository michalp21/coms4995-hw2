import tensorflow as tf
import numpy as np

from datetime import datetime

from pandas import DataFrame

from hw1_helpers import *

data_root_path = '/home/daniel/cifar10-hw2/'

BATCH_SIZE = 100
EVAL_BATCH_SIZE = 400
NUM_EXAMPLES = 50000
NUM_EVAL_EXAMPLES = 10000
NUM_TRAIN_EXAMPLES = NUM_EXAMPLES - NUM_EVAL_EXAMPLES

X_test = get_images(data_root_path + 'test')
X_test = X_test.T
X_test = tf.cast(X_test, tf.float32)
X_test = tf.reshape(X_test, [10000, 32, 32, 3])

# Get all the data and shape it for TF

X_all, y_all = get_train_data(data_root_path)

X_all = X_all.T
X_all = tf.cast(X_all, tf.float32)
X_all = tf.reshape(X_all, [50000, 32, 32, 3])

print('Data loading hw1 done: %s')

# Placeholder boolean tensor so we know whether we are training or eval/predicting
# Train means update the graph and use dropout
# Eval means no dropout

train_mode = tf.placeholder(dtype=tf.bool)
final_mode = tf.placeholder(dtype=tf.bool)

# Set up tensors to fetch the batch for training or eval. Which set of tensors
# is activated will depend on train_mode.

batch_train = tf.random_uniform([BATCH_SIZE], minval=NUM_EVAL_EXAMPLES,
	maxval=NUM_EXAMPLES, dtype=tf.int32)
batch_eval = tf.random_uniform([EVAL_BATCH_SIZE],
	minval=0, maxval=NUM_EVAL_EXAMPLES, dtype=tf.int32)

X_batch = tf.gather(X_all, batch_train)
y_batch = tf.gather(y_all, batch_train)

X_eval = tf.gather(X_all, batch_eval)
y_eval = tf.gather(y_all, batch_eval)

# Evaluate whether it's train or eval mode and decide on the inputs/labels.

inputs = tf.cond(train_mode, true_fn=lambda: X_batch, false_fn=lambda: X_eval)
inputs = tf.cond(final_mode, true_fn=lambda: X_test, false_fn=lambda: inputs)
labels = tf.cond(train_mode, true_fn=lambda: y_batch, false_fn=lambda: y_eval)
onehot = tf.one_hot(indices=labels, depth=10)

### Build the actual network structure ###

# Convolutional + Pooling Layers #1
net = tf.layers.conv2d(
    inputs=inputs,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

# Convolutional + Pooling Layers 2
net = tf.layers.conv2d(
    inputs=net,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

# Dense (fully connected) Layer
net = tf.reshape(net, [-1, 8 * 8 * 64])
net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)

# Dropout layer -- do not dropout for eval
net = tf.layers.dropout(
    inputs=net, rate=0.4, training=train_mode)

# Logits
logits = tf.layers.dense(inputs=net, units=10)

# train -- set up the gradient descent. Obviousy don't call for eval
loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot, logits=logits)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(
	loss=loss,
	global_step=tf.train.get_global_step())


# eval
predictions = tf.argmax(input=logits, axis=1)
diff = tf.to_float(tf.equal(predictions, labels))
accuracy = tf.reduce_mean(diff)

# Running

print("Try training")

sess = tf.Session()
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
tf.train.start_queue_runners(sess)

for i in range(50000):
	_, loss_value = sess.run([train_op, loss], feed_dict={final_mode: False, train_mode: True})
	print("%d train: %.4f" % (i, loss_value))

	if i % 50 == 0:
		accu = sess.run((accuracy), feed_dict={final_mode: False, train_mode: False})
		print("Accuracy: %.1f%%" % (100.0 * accu))

	if i % 1000 == 0:
		# Print out some real evals
		pred = sess.run((predictions), feed_dict={final_mode: True, train_mode: False})
		df = DataFrame(data=pred)
		df.index.name = 'ID'
		df.colums.values[0] = 'CLASS'
		filename = './pred-' + datetime.now().strftime('%d-%H:%M:%S' + '.txt')
		df.to_csv(filename, mode='a', index=True, sep=',')
		print("...saved to " + filename)

print("Done training")