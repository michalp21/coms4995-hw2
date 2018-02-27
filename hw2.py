import tensorflow as tf
import numpy as np

from datetime import datetime

from pandas import DataFrame

from hw1_helpers import *

data_root_path = '/home/daniel/cifar10-hw2/'

BATCH_SIZE = 128
NUM_EXAMPLES = 50000
NUM_EVAL_EXAMPLES = 512
NUM_TRAIN_EXAMPLES = NUM_EXAMPLES - NUM_EVAL_EXAMPLES

X_test = get_images(data_root_path + 'test', True)
X_test = X_test.T
X_test = tf.cast(X_test, tf.float32)
X_test = tf.reshape(X_test, [10000, 32, 32, 3])

# Get all the data and shape it for TF

X_all, y_all = get_train_data(data_root_path)

perm = np.random.permutation(NUM_EXAMPLES)
X_all = X_all[:, perm]
y_all = y_all[perm]

X_eval = X_all[:, 0:NUM_EVAL_EXAMPLES]
y_eval = y_all[0:NUM_EVAL_EXAMPLES]

X_all = X_all.T
X_all = tf.cast(X_all, tf.float32)
X_all = tf.reshape(X_all, [NUM_EXAMPLES, 32, 32, 3])

X_eval = X_eval.T
X_eval = tf.cast(X_eval, tf.float32)
X_eval = tf.reshape(X_eval, [NUM_EVAL_EXAMPLES, 32, 32, 3])

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

X_batch_train = tf.gather(X_all, batch_train)
y_batch_train = tf.gather(y_all, batch_train)

X_batch_train = tf.map_fn(
	lambda img: tf.image.random_flip_left_right(img), X_batch_train)

# Evaluate whether it's train or eval mode and decide on the inputs/labels.

inputs = tf.cond(train_mode, true_fn=lambda: X_batch_train, false_fn=lambda: X_eval)
inputs = tf.cond(final_mode, true_fn=lambda: X_test, false_fn=lambda: inputs)
labels = tf.cond(train_mode, true_fn=lambda: y_batch_train, false_fn=lambda: y_eval)
onehot = tf.one_hot(indices=labels, depth=10)

inputs = tf.map_fn(lambda img: tf.image.per_image_standardization(img), inputs)

### Build the actual network structure ###

# Convolutional + Pooling Layers #1
net = tf.layers.conv2d(
    inputs=inputs,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# Convolutional + Pooling Layers 2
net = tf.layers.conv2d(
    inputs=net,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

# Dense (fully connected) Layer
net = tf.reshape(net, [-1, 8 * 8 * 64])
net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)
net = tf.layers.dropout(
    inputs=net, rate=0.4, training=train_mode)
net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)
# Dropout layer -- do not dropout for eval
net = tf.layers.dropout(
    inputs=net, rate=0.4, training=train_mode)

# Logits
logits = tf.layers.dense(inputs=net, units=10)

# train -- set up the gradient descent. Obviousy don't call for eval
loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot, logits=logits)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(
	loss=loss,
	global_step=tf.train.get_global_step())


# eval
predictions = tf.argmax(input=logits, axis=1)
diff = tf.to_float(tf.equal(predictions, labels))
accuracy = tf.reduce_mean(diff)

# Running

print("Try training: left right flip and per image std")

sess = tf.Session()
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
tf.train.start_queue_runners(sess)

best_accu_p = 68.0

for i in range(500000):
	_, loss_value = sess.run([train_op, loss], feed_dict={final_mode: False,
		train_mode: True})
	print("%d train: %.1f" % (i, loss_value))

	# these numbers are just ballparks for how to make training convenient
	if i % 100 == 0 or i > 1000 and i % 60 == 0 or i > 4000 and i % 40 == 0:
		accu_p = 100 * sess.run((accuracy),
			feed_dict={final_mode: False, train_mode: False})
		print("[%s] Accuracy: %.1f%%" % (str(datetime.now()), accu_p))
		if accu_p > best_accu_p:
			filename = './accu-' + str(i) + '-' + ('%.2f' % (accu_p,)) + '.txt'
			print("New best accuracy %.2f%% (old %.2f%%), saving to %s"
				%  (accu_p, best_accu_p, filename))
			best_accu_p = accu_p

			pred = sess.run((predictions), feed_dict={final_mode: True,
				train_mode: False})
			df = DataFrame(data=pred)
			df.to_csv(filename, mode='a', index=True, sep=',')


	if i > 0 and i % 10000 == 0:
		# Print out some real evals
		filename = 'pred-%d.txt' % (i,)
		pred = sess.run((predictions), feed_dict={final_mode: True,
			train_mode: False})
		df = DataFrame(data=pred)
		df.to_csv(filename, mode='a', index=True, sep=',')

		# Manually add header before kaggle submission
		print("...saved every 10000th to " + filename)

print("Done training")