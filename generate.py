import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_var  = tf.Variable(tf.constant(0.0, shape=[200,784], dtype=tf.float32))
# x_image = tf.reshape(x_var, [200,28,28,1])

# W_conv1 = weight_variable([3, 3, 1, 16])
# b_conv1 = bias_variable([16])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# W_conv2 = weight_variable([3, 3, 16, 16])
# b_conv2 = bias_variable([16])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# # h_pool2 = max_pool_2x2(h_conv2)
#
#
# W_conv3 = weight_variable([3, 3, 16, 32])
# b_conv3 = bias_variable([32])
# h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
#
# W_conv4 = weight_variable([3, 3, 32, 64])
# b_conv4 = bias_variable([64])
# h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
# h_pool4 = max_pool_2x2(h_conv4)
#
#
W_fc1 = weight_variable([28 * 28, 1024])
b_fc1 = bias_variable([1024])

# h_pool4_flat = tf.reshape(h_pool4, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(x_var, W_fc1) + b_fc1)
#
keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 1024])
b_fc2 = bias_variable([1024])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc4 = weight_variable([1024, 1024])
b_fc4 = bias_variable([1024])

h_fc4 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc4) + b_fc4)
h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)


W_fc3 = weight_variable([1024, 10])
b_fc3 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc4_drop, W_fc3) + b_fc3)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), reduction_indices=[1]))
    tf.scalar_summary('cross entropy', cross_entropy)

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summary = tf.scalar_summary('accuracy', accuracy)


merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('./train', sess.graph)
test_writer = tf.train.SummaryWriter('./test')


objective = tf.gather(tf.gather(y_conv, 0), 2)


adverserial_update = tf.gradients(objective, [x_var])[0]


sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

for i in range(500):
    batch = mnist.train.next_batch(200)

    if i%100 == 0 and i!=0:
        print "step", i, "loss", loss_val

    data_assign = x_var.assign(batch[0])
    sess.run(data_assign)

    _, loss_val = sess.run([train_step, cross_entropy], feed_dict={y_: batch[1], keep_prob: 0.5})

save_path = saver.save(sess, "model.ckpt")

saver.restore(sess, "model.ckpt")
data_assign = x_var.assign(np.asarray(mnist.test.images[:200]).astype(float))
sess.run(data_assign)
print("test accuracy %g"%accuracy.eval(feed_dict={y_: mnist.test.labels[:200], keep_prob: 1.0}))

print mnist.test.labels[0]

# ob = sess.run([objective], feed_dict={y_: mnist.test.labels[:200], keep_prob: 1.0})
# print ob

for i in range(1):
    grad = sess.run([adverserial_update], feed_dict={y_: mnist.test.labels[:200], keep_prob: 1.0})
    # X = np.asarray(mnist.test.images[0]) + 0.0001*np.sign(grad[0][0])
    X = np.asarray(mnist.test.images[0]) + 1000000.0*grad[0][0]
    gg = sess.run([y_conv], feed_dict={y_: mnist.test.labels[:200], keep_prob: 1.0})
    X_modified = mnist.test.images[:200]
    X_modified[0] = X
    data_assign = x_var.assign(np.asarray(X_modified).astype(float))
    sess.run(data_assign)


gg = sess.run([y_conv], feed_dict={y_: mnist.test.labels[:200], keep_prob: 1.0})
print gg[0][0]

import cv2
x=X
x=x.reshape(28,28)
img = cv2.imshow("a",x)
k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
