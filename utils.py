import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from torch_geometric.datasets import Planetoid


def k_core_s(G):
    adj = np.array(nx.adjacency_matrix(G).todense())
    # 获取节点core数目列表
    core_num = nx.core_number(G)
    # 获取最大节点core数目列表
    max_core_num = max(list(core_num.values()))
    adj = adj.tolist()
    # 归一化
    for i in range(len(adj)):
        for j in range(len(adj)):
            if i > j:
                adj[i][j] = 0
            elif adj[i][j] != 0:
                adj[i][j] = (adj[i][j] * core_num[i]) / max_core_num
    adj = np.array(adj, dtype='float32')
    # 矩阵转置翻转
    sim = matrix_transpose_flip(adj)
    return sim


def matrix_transpose_flip(adj):
    adj_T = adj.T
    np.fill_diagonal(adj_T, 0)  # 将对角线清零
    adj_res = adj + adj_T
    return adj_res


def data_to_save(res, SAVE_PATH, columns):
    # 检查文件是否存在且不为空
    file_exists = os.path.isfile(SAVE_PATH)
    file_is_empty = not file_exists or os.path.getsize(SAVE_PATH) == 0

    #  保存实验结果
    if file_is_empty:
        data = pd.DataFrame(res, columns=columns)
        data.to_csv(SAVE_PATH, columns=columns, mode="a+", header=True, index=False)
    else:
        data = pd.DataFrame(res)
        data.to_csv(SAVE_PATH, mode="a+", header=False, index=False)


def get_dataset(dataset):
    datasets = Planetoid('./dataset', dataset)
    return datasets


def data_preprocessing(dataset):
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]),
        torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    # test_data = dataset.adj
    # test_data = dataset.adj_label

    dataset.adj += torch.eye(dataset.x.shape[0])
    # test_data = dataset.adj
    dataset.adj = normalize(dataset.adj, norm="l1")
    # test_data = dataset.adj
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)
    # test_data = dataset.adj

    return dataset


def data_preprocessing_new(dataset):
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]),
        torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    G = nx.from_numpy_array(dataset.adj.numpy())
    G.remove_edges_from(nx.selfloop_edges(G))
    sim = k_core_s(G)

    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    # G = nx.from_numpy_array(dataset.adj.numpy())
    # G.remove_edges_from(nx.selfloop_edges(G))
    # sim = k_core_s(G)

    dataset.sim = torch.from_numpy(sim).to(dtype=torch.float)

    return dataset


def data_preprocessing_ACM_DBLP(dataset, adj):
    dataset['adj'] = adj.to_dense()
    dataset['adj_label'] = adj

    G = nx.from_numpy_array(dataset['adj'].numpy())
    G.remove_edges_from(nx.selfloop_edges(G))
    sim = k_core_s(G)

    dataset['adj'] += torch.eye(dataset['adj'].shape[0])
    dataset['adj'] = normalize(dataset['adj'], norm="l1")
    dataset['adj'] = torch.from_numpy(dataset['adj']).to(dtype=torch.float)

    # G = nx.from_numpy_array(dataset.adj.numpy())
    # G.remove_edges_from(nx.selfloop_edges(G))
    # sim = k_core_s(G)

    dataset['sim'] = torch.from_numpy(sim).to(dtype=torch.float)

    return dataset


def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t = 2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)
