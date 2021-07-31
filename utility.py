from math import log
import numpy as np
import pandas as pd
import copy
from operator import itemgetter
import time
import tqdm

from scipy import stats
from scipy.sparse import coo_matrix
from multiprocessing import Process, Queue, Pool, Manager
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import pickle

top1 = 1
top2 = 5
top3 = 10
top4 = 20
k_set = [top1, top2, top3, top4]


def negative_sampling_AutoRec(num_row, num_col, row_array, col_array, neg):
    row = np.tile(row_array, neg + 1).reshape(row_array.shape[0] * (neg + 1))
    pos = col_array.reshape((col_array.shape[0], 1))
    neg = np.random.choice(np.arange(num_col), size=(neg * col_array.shape[0]),
                           replace=True).reshape((col_array.shape[0] * neg, 1))
    col = np.concatenate([pos, neg], axis=0)
    mask = coo_matrix((np.ones(row.shape[0]), (row, col.reshape(col.shape[0]))),
                      shape=(num_row, num_col)).toarray()
    return mask


def test_model(num_u, Rec, like, test_like, precision_queue, recall_queue, ndcg_queue, n_user_queue):
    precision = np.array([0.0, 0.0, 0.0, 0.0])
    recall = np.array([0.0, 0.0, 0.0, 0.0])
    ndcg = np.array([0.0, 0.0, 0.0, 0.0])

    user_num = num_u

    for i in range(num_u):
        Rec[i, like[i]] = -100000.0

    for u in range(num_u):  # iterate each user
        scores = Rec[u, :]
        top_iid = np.argpartition(scores, -20)[-20:]
        top_iid = top_iid[np.argsort(scores[top_iid])[-1::-1]]

        # calculate the metrics
        if not len(test_like[u]) == 0:
            precision_u, recall_u, ndcg_u = user_precision_recall_ndcg(top_iid, test_like[u])
            precision += precision_u
            recall += recall_u
            ndcg += ndcg_u
        else:
            user_num -= 1
    precision_queue.put(precision)
    recall_queue.put(recall)
    ndcg_queue.put(ndcg)
    n_user_queue.put(user_num)


def MP_test_model_all(Rec, test_like, train_like, n_workers=10):
    m = Manager()
    precision_queue = m.Queue(maxsize=n_workers)
    recall_queue = m.Queue(maxsize=n_workers)
    ndcg_queue = m.Queue(maxsize=n_workers)
    n_user_queue = m.Queue(maxsize=n_workers)
    processors = []

    num_user = Rec.shape[0]

    num_user_each = int(num_user / n_workers)
    for i in range(n_workers):
        if i < n_workers - 1:
            p = Process(target=test_model, args=(num_user_each,
                                                 Rec[num_user_each * i: num_user_each * (i + 1)],
                                                 train_like[num_user_each * i: num_user_each * (i + 1)],
                                                 test_like[num_user_each * i: num_user_each * (i + 1)],
                                                 precision_queue,
                                                 recall_queue,
                                                 ndcg_queue,
                                                 n_user_queue))
            processors.append(p)
        else:
            p = Process(target=test_model, args=(num_user - num_user_each * i,
                                                 Rec[num_user_each * i: num_user],
                                                 train_like[num_user_each * i: num_user],
                                                 test_like[num_user_each * i: num_user],
                                                 precision_queue,
                                                 recall_queue,
                                                 ndcg_queue,
                                                 n_user_queue))
            processors.append(p)
        p.start()
    print('!!!!!!!!!!!!!!!!!test start!!!!!!!!!!!!!!!!!!')

    for p in processors:
        p.join()
    precision = precision_queue.get()
    while not precision_queue.empty():
        tmp = precision_queue.get()
        precision += tmp
    recall = recall_queue.get()
    while not recall_queue.empty():
        tmp = recall_queue.get()
        recall += tmp
    ndcg = ndcg_queue.get()
    while not ndcg_queue.empty():
        tmp = ndcg_queue.get()
        ndcg += tmp
    n_user = n_user_queue.get()
    while not n_user_queue.empty():
        tmp = n_user_queue.get()
        n_user += tmp

    # compute the average over all users
    precision /= n_user
    recall /= n_user
    ndcg /= n_user

    print('precision_%d\t[%.7f],\t||\t precision_%d\t[%.7f],\t||\t precision_%d\t[%.7f],\t||\t precision_%d\t[%.7f]'
          % (k_set[0], precision[0], k_set[1], precision[1], k_set[2], precision[2], k_set[3], precision[3]))

    print('recall_%d   \t[%.7f],\t||\t recall_%d   \t[%.7f],\t||\t recall_%d   \t[%.7f],\t||\t recall_%d   \t[%.7f]'
          % (k_set[0], recall[0], k_set[1], recall[1], k_set[2], recall[2], k_set[3], recall[3]))

    f_measure_1 = 2 * (precision[0] * recall[0]) / (precision[0] + recall[0]) if not precision[0] + recall[0] == 0 else 0
    f_measure_5 = 2 * (precision[1] * recall[1]) / (precision[1] + recall[1]) if not precision[1] + recall[1] == 0 else 0
    f_measure_10 = 2 * (precision[2] * recall[2]) / (precision[2] + recall[2]) if not precision[2] + recall[2] == 0 else 0
    f_measure_15 = 2 * (precision[3] * recall[3]) / (precision[3] + recall[3]) if not precision[3] + recall[3] == 0 else 0
    print('f_measure_%d\t[%.7f],\t||\t f_measure_%d\t[%.7f],\t||\t f_measure_%d\t[%.7f],\t||\t f_measure_%d\t[%.7f]'
          % (k_set[0], f_measure_1, k_set[1], f_measure_5, k_set[2], f_measure_10, k_set[3], f_measure_15))
    f_score = np.array([f_measure_1, f_measure_5, f_measure_10, f_measure_15])

    print('ndcg_%d     \t[%.7f],\t||\t ndcg_%d     \t[%.7f],\t||\t ndcg_%d     \t[%.7f],\t||\t ndcg_%d     \t[%.7f]'
          % (k_set[0], ndcg[0], k_set[1], ndcg[1], k_set[2], ndcg[2], k_set[3], ndcg[3]))
    return precision, recall, f_score, ndcg


def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm


def relu(x):
    return np.maximum(x, 0)


# calculate NDCG@k
def NDCG_at_k(predicted_list, ground_truth, k):
    dcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(predicted_list[:k])]
    dcg = np.sum(dcg_value)
    if len(ground_truth) < k:
        ground_truth += [0 for i in range(k - len(ground_truth))]
    idcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(ground_truth[:k])]
    idcg = np.sum(idcg_value)
    return dcg / idcg


# calculate precision@k, recall@k, NDCG@k, where k = 1,5,10,15
def user_precision_recall_ndcg(new_user_prediction, test):
    dcg_list = []

    # compute the number of true positive items at top k
    count_1, count_5, count_10, count_15 = 0, 0, 0, 0
    for i in range(k_set[3]):
        if i < k_set[0] and new_user_prediction[i] in test:
            count_1 += 1.0
        if i < k_set[1] and new_user_prediction[i] in test:
            count_5 += 1.0
        if i < k_set[2] and new_user_prediction[i] in test:
            count_10 += 1.0
        if new_user_prediction[i] in test:
            count_15 += 1.0
            dcg_list.append(1)
        else:
            dcg_list.append(0)

    # calculate NDCG@k
    idcg_list = [1 for i in range(len(test))]
    ndcg_tmp_1 = NDCG_at_k(dcg_list, idcg_list, k_set[0])
    ndcg_tmp_5 = NDCG_at_k(dcg_list, idcg_list, k_set[1])
    ndcg_tmp_10 = NDCG_at_k(dcg_list, idcg_list, k_set[2])
    ndcg_tmp_15 = NDCG_at_k(dcg_list, idcg_list, k_set[3])

    # precision@k
    precision_1 = count_1 * 1.0 / k_set[0]
    precision_5 = count_5 * 1.0 / k_set[1]
    precision_10 = count_10 * 1.0 / k_set[2]
    precision_15 = count_15 * 1.0 / k_set[3]

    l = len(test)
    if l == 0:
        l = 1
    # recall@k
    recall_1 = count_1 / l
    recall_5 = count_5 / l
    recall_10 = count_10 / l
    recall_15 = count_15 / l

    # return precision, recall, ndcg_tmp
    return np.array([precision_1, precision_5, precision_10, precision_15]), \
           np.array([recall_1, recall_5, recall_10, recall_15]), \
           np.array([ndcg_tmp_1, ndcg_tmp_5, ndcg_tmp_10, ndcg_tmp_15])


def print_sorted_dict(dictionary):
    tmp = []
    for key, value in [(k, dictionary[k]) for k in sorted(dictionary, key=dictionary.get)]:
        tmp.append(value)
        print("# %s: %s" % (key, value))
    rstd = np.std(tmp) / np.mean(tmp)
    print("# relative std = " + str(rstd))
    return rstd
