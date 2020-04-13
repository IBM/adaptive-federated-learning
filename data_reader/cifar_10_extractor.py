import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from struct import *
import numpy as np
from util.utils import get_one_hot_from_label_index

BYTES_EACH_LABEL = 1
BYTES_EACH_IMAGE = 3072
BYTES_EACH_SAMPLE = BYTES_EACH_LABEL + BYTES_EACH_IMAGE
SAMPLES_EACH_FILE = 10000


def cifar_10_extract_samples(sample_list, is_train=True, file_path=os.path.dirname(__file__)):
    f_list = []

    if is_train:
        f_list.append(open(file_path + '/cifar-10-batches-bin/data_batch_1.bin', 'rb'))
        f_list.append(open(file_path + '/cifar-10-batches-bin/data_batch_2.bin', 'rb'))
        f_list.append(open(file_path + '/cifar-10-batches-bin/data_batch_3.bin', 'rb'))
        f_list.append(open(file_path + '/cifar-10-batches-bin/data_batch_4.bin', 'rb'))
        f_list.append(open(file_path + '/cifar-10-batches-bin/data_batch_5.bin', 'rb'))
    else:
        f_list.append(open(file_path + '/cifar-10-batches-bin/test_batch.bin', 'rb'))

    data = []
    labels = []

    for i in sample_list:
        file_index = int(i / float(SAMPLES_EACH_FILE))
        f_list[file_index].seek((i - file_index * SAMPLES_EACH_FILE) * BYTES_EACH_SAMPLE)
        label = unpack('>B', f_list[file_index].read(1))[0]
        y = get_one_hot_from_label_index(label)
        x = np.array(list(f_list[file_index].read(BYTES_EACH_IMAGE)))

        # Simple normalization (choose either this or next approach)
        # x = x / 255.0

        # Normalize according to mean and standard deviation as suggested in Tensorflow tutorial
        tmp_mean = np.mean(x)
        tmp_stddev = np.std(x)
        tmp_adjusted_stddev = max(tmp_stddev, 1.0 / np.sqrt(len(x)))
        x = (x - tmp_mean) / tmp_adjusted_stddev

        # Reshaping to match with Tensorflow format
        x = np.reshape(x, [32, 32, 3], order='F')
        x = np.reshape(x, [3072], order='C')

        data.append(x)
        labels.append(y)

    for f in f_list:
        f.close()

    return data, labels


def cifar_10_extract(start_sample_index, num_samples, is_train=True, file_path=os.path.dirname(__file__)):
    sample_list = range(start_sample_index, start_sample_index + num_samples)
    return cifar_10_extract_samples(sample_list, is_train, file_path)