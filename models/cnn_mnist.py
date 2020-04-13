import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from models.cnn_abstract import ModelCNNAbstract

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class ModelCNNMnist(ModelCNNAbstract):
    def __init__(self):
        super().__init__()
        pass

    def create_graph(self, learning_rate=None):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        self.W_conv2 = weight_variable([5, 5, 32, 32])
        self.b_conv2 = bias_variable([32])
        self.W_fc1 = weight_variable([7 * 7 * 32, 256])
        self.b_fc1 = bias_variable([256])
        self.W_fc2 = weight_variable([256, 10])
        self.b_fc2 = bias_variable([10])

        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)
        self.h_norm1 = tf.nn.lrn(self.h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        self.h_conv2 = tf.nn.relu(conv2d(self.h_norm1, self.W_conv2) + self.b_conv2)
        self.h_norm2 = tf.nn.lrn(self.h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        self.h_pool2 = max_pool_2x2(self.h_norm2)
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 32])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        self.y = tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

        self.all_weights = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
                            self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

        self._assignment_init()

        self._optimizer_init(learning_rate=learning_rate)
        self.grad = self.optimizer.compute_gradients(self.cross_entropy, var_list=self.all_weights)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self._session_init()
        self.graph_created = True
