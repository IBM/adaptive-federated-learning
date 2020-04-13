import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
import abc

LOSS_ACC_BATCH_SIZE = 100  # When computing loss and accuracy, use blocks of LOSS_ACC_BATCH_SIZE


class ModelCNNAbstract(abc.ABC):
    def __init__(self):
        self.graph_created = False
        pass

    @abc.abstractmethod
    def create_graph(self, learning_rate=None):
        # The below variables need to be defined in the child class
        self.all_weights = None
        self.x = None
        self.y_ = None
        self.y = None
        self.cross_entropy = None
        self.acc = None

        self.init = None
        self.all_assignment_placeholders = None
        self.all_assignment_operations = None

        self._optimizer_init(learning_rate=learning_rate)
        self.grad = None

        self.session = None  # Used for consecutive training

    def _optimizer_init(self, learning_rate=None):
        if learning_rate is None:
            learning_rate = 0.0   # The learning rate should not have effect when not using optimizer
        self.learning_rate = learning_rate
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.optimizer_op = self.optimizer.minimize(self.cross_entropy)

    def _assignment_init(self):
        self.init = tf.global_variables_initializer()

        self.all_assignment_placeholders = []
        self.all_assignment_operations = []
        for w in self.all_weights:
            p = tf.placeholder(tf.float32, shape=w.get_shape())
            self.all_assignment_placeholders.append(p)
            self.all_assignment_operations.append(w.assign(p))

    def _session_init(self):
        self.session = tf.Session()

    def get_weight_dimension(self, imgs, labels):
        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        dim = 0

        for weight in self.all_weights:
            tmp = 1
            l = weight.get_shape()
            for i in range(0, len(l)):
                tmp *= l[i].value

            dim += tmp

        return dim

    def get_init_weight(self, dim, rand_seed=None):
        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        if rand_seed is not None:
            # Random seed only works at graph initialization, so recreate graph here
            self.session.close()
            tf.reset_default_graph()
            tf.set_random_seed(rand_seed)
            self.create_graph(learning_rate=self.learning_rate)  # This creates the session as well

        self.session.run(self.init)

        weight_flatten_list = []
        for weight in self.all_weights:
            weight_var = self.session.run(weight)
            weight_flatten_list.append(np.reshape(weight_var, weight_var.size))

        weight_flatten_array = np.hstack(weight_flatten_list)

        return weight_flatten_array

    def assign_flattened_weight(self, sess, w):
        start_index = 0

        for k in range(0, len(self.all_weights)):
            weight = self.all_weights[k]

            tmp = 1
            l = weight.get_shape()
            for i in range(0, len(l)):
                tmp *= l[i].value

            weight_var = np.reshape(w[start_index : start_index+tmp], l)
            sess.run(self.all_assignment_operations[k], feed_dict={self.all_assignment_placeholders[k]: weight_var})

            del weight_var

            start_index = start_index + tmp

    def gradient(self, imgs, labels, w, sample_indices):
        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        self.assign_flattened_weight(self.session, w)

        grad_var_list = self.session.run(self.grad, feed_dict={self.x: [imgs[i] for i in sample_indices], self.y_: [labels[i] for i in sample_indices]})

        grad_flatten_list = []
        for l in grad_var_list:
            grad_flatten_list.append(np.reshape(l[0], l[0].size))

        grad_flatten_array = np.hstack(grad_flatten_list)

        del grad_var_list
        del grad_flatten_list

        return grad_flatten_array

    def loss(self, imgs, labels, w, sample_indices=None):
        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        self.assign_flattened_weight(self.session, w)

        if sample_indices is None:
            sample_indices = range(0, len(labels))

        val = 0
        l = []
        for k in range(0, len(sample_indices)):
            l.append(sample_indices[k])

            if len(l) >= LOSS_ACC_BATCH_SIZE or k == len(sample_indices) - 1:
                val += self.session.run(self.cross_entropy,
                                     feed_dict={self.x: [imgs[i] for i in l],
                                                self.y_: [labels[i] for i in l]}) \
                       * float(len(l)) / len(sample_indices)

                l = []

        return val

    def accuracy(self, imgs, labels, w, sample_indices=None):
        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        self.assign_flattened_weight(self.session, w)

        if sample_indices is None:
            sample_indices = range(0, len(labels))

        val = 0
        l = []
        for k in range(0, len(sample_indices)):
            l.append(sample_indices[k])

            if len(l) >= LOSS_ACC_BATCH_SIZE or k == len(sample_indices) - 1:

                val += self.session.run(self.acc, feed_dict={self.x: [imgs[i] for i in l],
                                                             self.y_: [labels[i] for i in l]}) \
                       * float(len(l)) / len(sample_indices)

                l = []

        return val

    def start_consecutive_training(self, w_init):
        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        self.assign_flattened_weight(self.session, w_init)

    def end_consecutive_training_and_get_weights(self):
        weight_flatten_list = []
        for weight in self.all_weights:
            weight_var = self.session.run(weight)
            weight_flatten_list.append(np.reshape(weight_var, weight_var.size))

        weight_flatten_array = np.hstack(weight_flatten_list)

        return weight_flatten_array

    def run_one_step_consecutive_training(self, imgs, labels, sample_indices):
        self.session.run(self.optimizer_op, feed_dict={self.x: [imgs[i] for i in sample_indices], self.y_: [labels[i] for i in sample_indices]})

    def predict(self, img, w):
        if not self.graph_created:
            raise Exception('Graph is not created. Call create_graph() first.')

        self.assign_flattened_weight(self.session, w)
        pred = self.session.run(self.y, feed_dict={self.x: [img]})
        return pred[0]   # self.y gives an array of predictions
