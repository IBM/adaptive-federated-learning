import numpy as np
import pickle, struct, socket, math


def get_even_odd_from_one_hot_label(label):
    for i in range(0, len(label)):
        if label[i] == 1:
            c = i % 2
            if c == 0:
                c = 1
            elif c == 1:
                c = -1
            return c


def get_index_from_one_hot_label(label):
    for i in range(0, len(label)):
        if label[i] == 1:
            return [i]


def get_one_hot_from_label_index(label, number_of_labels=10):
    one_hot = np.zeros(number_of_labels)
    one_hot[label] = 1
    return one_hot


def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())


def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    print(msg[0], 'received from', sock.getpeername())

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg


def moving_average(param_mvavr, param_new, movingAverageHoldingParam):
    if param_mvavr is None or np.isnan(param_mvavr):
        param_mvavr = param_new
    else:
        if not np.isnan(param_new):
            param_mvavr = movingAverageHoldingParam * param_mvavr + (1 - movingAverageHoldingParam) * param_new
    return param_mvavr


def get_indices_each_node_case(n_nodes, maxCase, label_list):
    indices_each_node_case = []

    for i in range(0, maxCase):
        indices_each_node_case.append([])

    for i in range(0, n_nodes):
        for j in range(0, maxCase):
            indices_each_node_case[j].append([])

    # indices_each_node_case is a big list that contains N-number of sublists. Sublist n contains the indices that should be assigned to node n

    min_label = min(label_list)
    max_label = max(label_list)
    num_labels = max_label - min_label + 1

    for i in range(0, len(label_list)):
        # case 1
        indices_each_node_case[0][(i % n_nodes)].append(i)

        # case 2
        tmp_target_node = int((label_list[i] - min_label) % n_nodes)
        if n_nodes > num_labels:
            tmp_min_index = 0
            tmp_min_val = math.inf
            for n in range(0, n_nodes):
                if n % num_labels == tmp_target_node and len(indices_each_node_case[1][n]) < tmp_min_val:
                    tmp_min_val = len(indices_each_node_case[1][n])
                    tmp_min_index = n
            tmp_target_node = tmp_min_index
        indices_each_node_case[1][tmp_target_node].append(i)

        # case 3
        for n in range(0, n_nodes):
            indices_each_node_case[2][n].append(i)

        # case 4
        tmp = int(np.ceil(min(n_nodes, num_labels) / 2))
        if label_list[i] < (min_label + max_label) / 2:
            tmp_target_node = i % tmp
        elif n_nodes > 1:
            tmp_target_node = int(((label_list[i] - min_label) % (min(n_nodes, num_labels) - tmp)) + tmp)

        if n_nodes > num_labels:
            tmp_min_index = 0
            tmp_min_val = math.inf
            for n in range(0, n_nodes):
                if n % num_labels == tmp_target_node and len(indices_each_node_case[3][n]) < tmp_min_val:
                    tmp_min_val = len(indices_each_node_case[3][n])
                    tmp_min_index = n
            tmp_target_node = tmp_min_index

        indices_each_node_case[3][tmp_target_node].append(i)

    return indices_each_node_case
