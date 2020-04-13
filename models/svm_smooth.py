import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from numpy import linalg


class ModelSVMSmooth:
    def __init__(self):
        self.lam = 0.1
        self.inner_prod_times_label = None
        self.w = None

    def get_weight_dimension(self, imgs, labels):

        return len(imgs[0])  # Assuming all images have the same size

    def get_init_weight(self, dim, rand_seed=None):
        return np.zeros(dim)

    # labels should be 1 or -1
    def gradient(self, imgs, labels, w, sampleIndices):
        val = 0

        self.w = w
        self.inner_prod_times_label = []

        for i in sampleIndices:

            tmp_inner_prod_times_label = labels[i] * np.inner(w,imgs[i])
            self.inner_prod_times_label.append(tmp_inner_prod_times_label)

            if tmp_inner_prod_times_label < 1.0:
                val = val - labels[i] * imgs[i] * (1 - tmp_inner_prod_times_label)

        val = self.lam * w + val / len(sampleIndices)
        return val

    def loss(self, imgs, labels, w, sample_indices = None):
        val = 0
        if sample_indices is None:
            sample_indices = range(0, len(labels))

        for i in sample_indices:
            val = val + pow(max(0.0, 1 - labels[i] * np.inner(w, imgs[i])), 2)

        val = 0.5 * self.lam * pow(linalg.norm(w), 2) + 0.5 * val / len(sample_indices)

        return val

    def loss_from_prev_gradient_computation(self):
        if (self.inner_prod_times_label is None) or (self.w is None):
            raise Exception('No previous gradient computation exists')

        val = 0
        for i in range(0, len(self.inner_prod_times_label)):
            val = val + pow(max(0.0, 1 - self.inner_prod_times_label[i]), 2)

        val = 0.5 * self.lam * pow(linalg.norm(self.w), 2) + 0.5 * val / len(self.inner_prod_times_label)
        return val

    def accuracy(self, imgs, labels, w):
        val = 0
        for i in range(1, len(labels)):
            if labels[i] * np.inner(w, imgs[i]) > 0:
                val += 1
        val /= len(labels)

        return val