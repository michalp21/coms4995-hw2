import tensorflow as tf
import numpy as np
import os

from datetime import datetime

from pandas import DataFrame

from hw1_helpers import *

data_root_path = 'cifar10-hw2/'

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

X_all = X_all.T
X_all = tf.cast(X_all, tf.float32)
X_all = tf.reshape(X_all, [NUM_EXAMPLES, 32, 32, 3])

print('Data loading hw1 done')

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

batch_eval = tf.range(NUM_EVAL_EXAMPLES)
X_eval = tf.gather(X_all, batch_eval)
y_eval = tf.gather(y_all, batch_eval)

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
    activation=tf.nn.relu,
    name = "conv0")
weights_conv0 = tf.get_default_graph().get_tensor_by_name(
  os.path.split(net.name)[0] + '/kernel:0')
tf.summary.histogram('conv0_weights', weights_conv0)

net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

net = tf.nn.lrn(net, 2, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# Convolutional + Pooling Layers 2
net = tf.layers.conv2d(
    inputs=net,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    name = "conv1")
weights_conv1 = tf.get_default_graph().get_tensor_by_name(
  os.path.split(net.name)[0] + '/kernel:0')
tf.summary.histogram('conv1_weights', weights_conv1)

net = tf.nn.lrn(net, 2, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

# Convolutional + Pooling Layers 3
net = tf.layers.conv2d(
    inputs=net,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    name = "conv2")
weights_conv2 = tf.get_default_graph().get_tensor_by_name(
  os.path.split(net.name)[0] + '/kernel:0')
tf.summary.histogram('conv2_weights', weights_conv2)

net = tf.nn.lrn(net, 2, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)



# Dense (fully connected) Layer
net = tf.reshape(net, [-1, 4 * 4 * 64])
net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu, name = "dense0",
                      kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1))
net = tf.layers.dropout(
    inputs=net, rate=0.4, training=train_mode)
net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu, name = "dense1",
                      kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1))
weights_dense1 = tf.get_default_graph().get_tensor_by_name(
  os.path.split(net.name)[0] + '/kernel:0')
tf.summary.histogram('dense_weights', weights_dense1)

# Dropout layer -- do not dropout for eval
net = tf.layers.dropout(
    inputs=net, rate=0.4, training=train_mode)

# Logits
logits = tf.layers.dense(inputs=net, units=10)

# train -- set up the gradient descent. Obviousy don't call for eval
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
baseloss = tf.losses.softmax_cross_entropy(onehot_labels=onehot, logits=logits)
loss = tf.add_n([baseloss] + reg_losses)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step())
tf.summary.scalar('loss', loss)

# eval
predictions = tf.argmax(input=logits, axis=1)
diff = tf.to_float(tf.equal(predictions, labels))
accuracy = tf.reduce_mean(diff)
tf.summary.scalar('accuracy', accuracy)

# Running

print("Try training")

sess = tf.Session()
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
tf.train.start_queue_runners(sess)

best_accu_p = 0.0


merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test')
tf.global_variables_initializer().run(session=sess)
for i in range(7000):
    _, loss_value, summary = sess.run([train_op, loss, merged], feed_dict={final_mode: False, train_mode: True})
    train_writer.add_summary(summary, i)
    print("%d train: %.1f" % (i, loss_value))

    # these numbers are just ballparks for how to make training convenient
    if i > 0 and (
        i % 100 == 0 or i > 1000 and i % 50 == 0 or i > 6000 and i % 25 == 0):
        accu_p = 100 * sess.run(accuracy, feed_dict={final_mode: False, train_mode: False})
        print("[%s] Accuracy: %.1f%%" % (str(datetime.now()), accu_p))
        if accu_p > best_accu_p:
            filename = './accu-' + str(i) + '-' + ('%.2f' % (accu_p,)) + '.txt'
            print("New best accuracy %.2f%% (old %.2f%%), saving to %s"
                %  (accu_p, best_accu_p, filename))
            best_accu_p = accu_p

            pred = sess.run((predictions), feed_dict={final_mode: True, train_mode: False})
            df = DataFrame(data=pred)
            df.index.name = 'ID'
            df.columns = ['CLASS']
            df.to_csv(filename, mode='a', index=True, sep=',')

train_writer.close()

print("Done training, submit: " + filename)