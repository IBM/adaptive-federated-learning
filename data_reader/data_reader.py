import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.utils import get_index_from_one_hot_label, get_even_odd_from_one_hot_label

def get_data(dataset, total_data, dataset_file_path=os.path.dirname(__file__), sim_round=None):

    if dataset == 'MNIST_ORIG_EVEN_ODD' or dataset == 'MNIST_ORIG_ALL_LABELS':
        from data_reader.mnist_extractor import mnist_extract

        if total_data > 60000:
            total_data_train = 60000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data

        if sim_round is None:
            start_index_train = 0
            start_index_test = 0
        else:
            start_index_train = (sim_round * total_data_train) % (max(1, 60000 - total_data_train + 1))
            start_index_test = (sim_round * total_data_test) % (max(1, 10000 - total_data_test + 1))

        train_image, train_label = mnist_extract(start_index_train, total_data_train, True, dataset_file_path)
        test_image, test_label = mnist_extract(start_index_test, total_data_test, False, dataset_file_path)

        # train_label_orig must be determined before the values in train_label are overwritten below
        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label[0])

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(train_label)):
                train_label[i] = get_even_odd_from_one_hot_label(train_label[i])

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(test_label)):
                test_label[i] = get_even_odd_from_one_hot_label(test_label[i])

    elif dataset == 'CIFAR_10':
        from data_reader.cifar_10_extractor import cifar_10_extract

        if total_data > 50000:
            total_data_train = 50000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data

        train_image, train_label = cifar_10_extract(0, total_data_train, True, dataset_file_path)
        test_image, test_label = cifar_10_extract(0, total_data_test, False, dataset_file_path)

        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label[0])

    else:
        raise Exception('Unknown dataset name.')

    return train_image, train_label, test_image, test_label, train_label_orig


def get_data_train_samples(dataset, samples_list, dataset_file_path=os.path.dirname(__file__)):
    if dataset == 'MNIST_ORIG_EVEN_ODD' or dataset == 'MNIST_ORIG_ALL_LABELS':
        from data_reader.mnist_extractor import mnist_extract_samples

        train_image, train_label = mnist_extract_samples(samples_list, True, dataset_file_path)

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(train_label)):
                train_label[i] = get_even_odd_from_one_hot_label(train_label[i])

    elif dataset == 'CIFAR_10':
        from data_reader.cifar_10_extractor import cifar_10_extract_samples

        train_image, train_label = cifar_10_extract_samples(samples_list, True, dataset_file_path)

    else:
        raise Exception('Training data sampling not supported for the given dataset name, use entire dataset by setting batch_size = total_data, ' +
                        'also confirm that dataset name is correct.')

    return train_image, train_label
