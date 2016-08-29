import tensorflow as tf
import os
import numpy as np
from corrosion_read import read_data 


# Initializations - Introduce small amount of noise for symmetry breaking and to prevent 0 gradients
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Slight positive bias prevents dead neurons
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Tensor flow session variable
sess = tf.InteractiveSession()

# Read in corrosion data
corrosion_train = read_data(os.path.join('data', 'train'), mode='train')
corrosion_test = read_data(os.path.join('data', 'test'), mode='test')

# Input images x will have 64x64x3=12288 size and batch size can be any size (None)
x = tf.placeholder(tf.float32, shape=[None, 12288])

# Number of classes, corroded or not corroded
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# Weights definition
#W = tf.Variable(tf.zeros([4096,2]))

# Biases definition
#b = tf.Variable(tf.zeros([2]))

#sess.run(tf.initialize_all_variables())


# 5x5 convolutions, 3 channels, 32 features
W_conv1 = weight_variable([5, 5, 3, 32])

b_conv1 = bias_variable([32])

# 64x64 images with 3 channels
x_image = tf.reshape(x, [-1,64,64,3])

# ReLU function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 2x2 Max pooling
h_pool1 = max_pool_2x2(h_conv1)

# 5x5 convolution on 32 input features with 64 output features
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

# ReLU function
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 2x2 max pooling
h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer, input is 16x16x64 now, 1024 neurons used
W_fc1 = weight_variable([16*16*64, 1024])
b_fc1 = bias_variable([1024])

# reshape into a vector
h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# placeholder for the probability that neurons output will be kept during dropout
keep_prob = tf.placeholder(tf.float32)

# Dropout layer for overfitting
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax layer
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# Train and evaluate the model

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = corrosion_train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i, train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        # TODO set the correct test set for corrosion dataset
        print "test accuracy %g"%accuracy.eval(feed_dict={
            x: corrosion_test.images, y_: corrosion_test.labels, keep_prob: 1.0})

