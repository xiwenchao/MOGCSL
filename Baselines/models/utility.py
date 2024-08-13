import os
import numpy as np
import pandas as pd
import torch
import time
from scipy.optimize import minimize
from scipy.optimize import nnls


def calculate_hit(predicts_dict,topk,labels_dict,hit_clicks,ndcg_clicks,hit_purchase,ndcg_purchase,task_num):  
    for task in range(task_num):
        sorted_list = np.argsort(predicts_dict[task])
        true_items = labels_dict[task]
        for i in range(len(topk)):
            rec_list = sorted_list[:, -topk[i]:]
            for j in range(len(true_items)):
                if true_items[j] in rec_list[j]:
                    rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                    if task == 0:
                        if predicts_dict[1][j] != -1:
                            hit_clicks[i] += 1.0
                            ndcg_clicks[i] += 1.0 / np.log2(rank + 1)
                    else:
                        hit_purchase[i] += 1.0
                        ndcg_purchase[i] += 1.0 / np.log2(rank + 1)

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist
    
"""
def calculate_hit(predicts_dict,topk,labels_dict,hit_click,ndcg_click,hit_purchase,ndcg_purchase,task_num):
    for task in range(task_num):
        sorted_list = np.argsort(predicts_dict[task])
        if len(sorted_list.shape) == 0:
            continue
        true_items = labels_dict[task]
        for i in range(len(topk)):
            rec_list = sorted_list[:, -topk[i]:]
            for j in range(len(true_items)):
                if true_items[j] in rec_list[j]:
                    rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                    if task == 0:
                        hit_click[i] += 1.0
                        ndcg_click[i] += 1.0 / np.log2(rank + 1)
                    else:
                        hit_purchase[i] += 1.0
                        ndcg_purchase[i] += 1.0 / np.log2(rank + 1)
"""


def extract_axis_1(data, ind, device):
    """
    Get specified elements along the first axis of tensor.
    :param data: PyTorch tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = torch.arange(data.size(0)).to(device)
    indices = torch.stack([batch_range, ind], dim=1)
    res = data[indices[:, 0], indices[:, 1]]

    return res


def pareto_step(w_a, w_b, c_a, c_b, paras):
    """
    Pareto step for two objectives.
    :param weights: Weights for two objectives.
    :param np.mat([[c_a], [c_b]]): Cost vector for two objectives.
    :param paras: Parameters for the step.
    :return: Updated weights.
    """
    w = np.mat([[w_a], [w_b]])
    c = np.mat([[c_a], [c_b]])
    G = paras
    GGT = np.matmul(G, np.transpose(G))  # (K, K)
    e = np.mat(np.ones(np.shape(w)))  # (K, 1)
    m_up = np.hstack((GGT, e))  # (K, K+1)
    m_down = np.hstack((np.transpose(e), np.mat(np.zeros((1, 1)))))  # (1, K+1)
    M = np.vstack((m_up, m_down))  # (K+1, K+1)
    z = np.vstack((-np.matmul(GGT, c), 1 - np.sum(c)))  # (K+1, 1)
    # hat_w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(M), M)), M), z)  # (K+1, 1)
    hat_w = np.matmul(np.linalg.pinv(M), z)  # (K+1, 1)
    hat_w = hat_w[:-1]  # (K, 1)
    hat_w = np.reshape(np.array(hat_w), (hat_w.shape[0],))  # (K,)
    c = np.reshape(np.array(c), (c.shape[0],))  # (K,)
    new_w = ASM(hat_w, c)
    return new_w

def ASM(hat_w, c):
    """
    ref:
    http://ofey.me/papers/Pareto.pdf,
    https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function-solution-so-that-it-sums-to-1
    :param hat_w: # (K,)
    :param c: # (K,)
    :return:
    """
    A = np.array([[0 if i != j else 1 for i in range(len(c))] for j in range(len(c))])
    b = hat_w
    x0, _ = nnls(A, b)

    def _fn(x, A, b):
        return np.linalg.norm(A.dot(x) - b)

    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) + np.sum(c) - 1}
    bounds = [[0., None] for _ in range(len(hat_w))]
    min_out = minimize(_fn, x0, args=(A, b), method='SLSQP', bounds=bounds, constraints=cons)
    new_w = min_out.x + c
    return new_w