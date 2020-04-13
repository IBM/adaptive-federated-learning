import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from struct import *
import numpy as np
from util.utils import get_one_hot_from_label_index


def mnist_extract_samples(sample_list, is_train=True, file_path=os.path.dirname(__file__)):
    file_path_extended = file_path + '/mnist'

    if is_train:
        f_images = open(file_path_extended + '/train-images.idx3-ubyte', 'rb')
        f_labels = open(file_path_extended + '/train-labels.idx1-ubyte', 'rb')
    else:
        f_images = open(file_path_extended + '/t10k-images.idx3-ubyte', 'rb')
        f_labels = open(file_path_extended + '/t10k-labels.idx1-ubyte', 'rb')

    s1, s2, s3, s4 = f_images.read(4), f_images.read(4), f_images.read(4), f_images.read(4)
    mn_im = unpack('>I', s1)[0]
    num_im = unpack('>I', s2)[0]
    rows_im = unpack('>I', s3)[0]
    cols_im = unpack('>I', s4)[0]

    mn_l = unpack('>I', f_labels.read(4))[0]
    num_l = unpack('>I', f_labels.read(4))[0]

    data = []
    labels = []

    for sample in sample_list:
        f_images.seek(16 + sample * rows_im * cols_im)
        f_labels.seek(8 + sample)

        x = np.array(list(f_images.read(rows_im * cols_im)))/255.0

        label = unpack('>B', f_labels.read(1))[0]
        y = get_one_hot_from_label_index(label)

        data.append(x)
        labels.append(y)

    f_images.close()
    f_labels.close()

    return data, labels


def mnist_extract(start_sample_index, num_samples, is_train=True, file_path=os.path.dirname(__file__)):
    sample_list = range(start_sample_index, start_sample_index + num_samples)
    return mnist_extract_samples(sample_list, is_train, file_path)
