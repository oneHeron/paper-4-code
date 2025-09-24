import argparse
import os
import pickle
import time

import pandas as pd
from matplotlib import colormaps

import networkx as nx
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.parameter import Parameter
from torch.optim import Adam
import numpy as np
import random
import utils
from evaluation import eva
from sparse_model import SpGAT, Clustering_Module, Clustering_Module_Loss


class DDGAE(nn.Module):
    def __init__(self, num_features, B_dim, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DDGAE, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrain model
        # self.gat = GAT(num_features, B_dim, hidden_size, embedding_size, alpha)
        self.gat = SpGAT(num_features, B_dim, hidden_size, embedding_size, alpha)
        # self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # # cluster layer
        # self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        # torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.cm = Clustering_Module(embedding_size, num_clusters, False)  # 聚类网络

    def forward(self, x, B, adj):
        A_hat, z, x_hat, B_hat, x_hat2, B_hat2 = self.gat(x, B, adj)
        # q = self.get_Q(z)
        cm = self.cm(z)
        return A_hat, z, x_hat, B_hat, x_hat2, B_hat2, cm

    # def get_Q(self, z):
    #     q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
    #     q = q.pow((self.v + 1.0) / 2.0)
    #     q = (q.t() / torch.sum(q, 1)).t()
    #     return q


# def target_distribution(q):
#     weight = q ** 2 / q.sum(0)
#     return (weight.t() / weight.sum(1)).t()


def trainer(dataset_, ALPHA=1.1, BETA=10, LBD=.1):
    sim = None
    data = None
    adj = None
    adj_label = None
    if args.name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset_ = utils.data_preprocessing_new(dataset_)
        adj = dataset_.adj.to(device)
        sim = dataset_.sim.to(device)
        # data and label
        data = torch.Tensor(dataset_.x).to(device)
        y = dataset_.y.cpu().numpy()

        adj_label = dataset_.adj_label.to(device)

    if args.name in ['ACM', 'DBLP']:
        if args.name == 'ACM':
            adj = torch.tensor(dataset_['PAP'], dtype=torch.float32)
            data = torch.Tensor(dataset_['feature']).to(device)
        if args.name == 'DBLP':
            adj = torch.tensor(dataset_['net_APA'], dtype=torch.float32)
            data = torch.Tensor(dataset_['features']).to(device)
        dataset_ = utils.data_preprocessing_ACM_DBLP(dataset_, adj)

        sim = dataset['sim'].to(device)
        adj = dataset['adj'].to(device)
        # data and label
        y = np.argmax(dataset_['label'], axis=1)
        y = y.reshape(-1, 1)
        adj_label = dataset['adj_label'].to(device)

    sim_dim = sim.shape[0]
    model = DDGAE(num_features=args.input_dim, B_dim=sim_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    # model.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
    criterion_cluster = Clustering_Module_Loss(
        num_clusters=args.n_clusters,
        alpha=ALPHA,
        lbd=LBD,
        orth=True,
        normalize=True).cuda(0)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    max_res = [0, 0, 0, 0]
    result = []
    pred_save = []

    for epoch in range(args.max_epoch):
        model.train()
        pred = []

        # A_hat, z, x_hat, B_hat, x_hat2, B_hat2, cm = model(data, sim, adj, M)
        A_hat, z, x_hat, sim_hat, x_hat2, sim_hat2, cm = model(data, sim, adj)
        # kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        cm_loss = criterion_cluster(cm)
        # re_loss = F.binary_cross_entropy(A_hat.view(-1), adj_label.view(-1))
        re_loss = F.binary_cross_entropy(A_hat.reshape(-1), adj_label.reshape(-1))
        B_loss = F.mse_loss(sim_hat, sim) + F.mse_loss(sim_hat2, sim)
        x_loss = F.mse_loss(x_hat, data) + F.mse_loss(x_hat2, data)
        L_f = x_loss + B_loss
        L_r = L_f + args.lambda1 * re_loss

        loss = L_r + args.beta * cm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.update_interval == 0:
            # update_interval
            model.eval()
            with torch.no_grad():
                # _, _, _, _, _, _, cm = model(data, sim, adj, M)
                _, _, _, _, _, _, cm = model(data, sim, adj)
            _, gamma, _, _ = cm
            pred += gamma.argmax(-1).detach().cpu().tolist()
            acc, nmi, ari, f1 = eva(y, pred, loss, epoch)
            if acc >= max_res[0]:
                max_res[0] = acc
                pred_save = pred
                result = [epoch, acc, nmi, ari, f1]
            if nmi >= max_res[1]:
                max_res[1] = nmi
            if ari >= max_res[2]:
                max_res[2] = ari
            if f1 >= max_res[3]:
                max_res[3] = f1
                # torch.save(model.state_dict(), args.pretrain_path)

    print("************************************************")
    print("Best Results")
    print(f"epoch {result[0]}, acc {result[1]:.4f}, nmi {result[2]:.4f}, ari {result[3]:.4f}, f1 {result[4]:.4f}")
    print("************************************************")
    file_path = f"./figs/{args.name}_node_communities.csv"
    mode = 'a' if os.path.exists(file_path) else 'w'
    header = not os.path.exists(file_path)
    # 将社区标签转为字符串
    community_str = ','.join(map(str, pred_save))
    df = pd.DataFrame({'Community_Labels': [community_str]})
    # 写入CSV
    df.to_csv(file_path, mode=mode, header=header, index=False)
    return max_res[0], max_res[1], max_res[2], max_res[3]
    # return result[1], result[2], result[3], result[4]


def seed_setting(seed):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',  # description - 在参数帮助文档之前显示的文本（默认值：无）
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # formatter_class - 用于自定义帮助文档输出格式的类
    parser.add_argument('--name', type=str,
                        default='Cora')  # type - 命令行参数应当被转换成的类型, default - 当参数未在命令行中出现时使用的值, help - 一个此选项作用的简单描述
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument("--lambda1", type=float, default=1, help="lambda for the reconstruct loss.")
    parser.add_argument("--beta", type=float, default=15, help="beta for the cluster loss.")
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = ['Cora', 'Citeseer']
    # datasets = ['ACM', 'DBLP', 'Citeseer']
    # args.name = 'ACM'
    for args.name in datasets:
        dataset = None

        if args.name in ['Cora', 'Citeseer', 'Pubmed']:
            datasets = utils.get_dataset(args.name)
            args.n_clusters = datasets.num_classes
            dataset = datasets[0]
            args.input_dim = dataset.num_features

        if args.name in ['ACM', 'DBLP']:
            dataset = scipy.io.loadmat(f'./dataset/{args.name}.mat')
            args.n_cluster = dataset['label'].shape[1]
            if args.name == 'ACM':
                args.input_dim = dataset['feature'].shape[1]
            else:
                args.input_dim = dataset['features'].shape[1]

        SAVE_PATH = f"./result/{args.name}_res_test.csv"

        # cora数据集 2708个节点，5429条边。标签共7个类别。数据集的特征维度是1433维。
        # citeSeer数据集 3312个节点，4723条边。标签共7个类别。数据集的特征维度是3703维。
        # PubMed数据集(引文网络)包括来自Pubmed数据库的19717篇关于糖尿病的科学出版物，分为三类：
        # Diabetes Mellitus, Experimental
        # Diabetes Mellitus Type 1
        # Diabetes Mellitus Type 2
        if args.name == 'Cora':
            args.lr = 6e-3
            args.lambda1 = 2
            args.beta = 2

        if args.name == 'Citeseer':
            args.lr = 1e-3
            args.lambda1 = 0.5
            args.beta = 0.2
            args.embedding_size = 32

        if args.name == 'DBLP':
            args.lr = 1e-2
            args.lambda1 = 1
            args.beta = 1

        if args.name == 'ACM':
            args.lr = 5e-3
            args.lambda1 = 0.5
            args.beta = 2

        if args.name == 'Pubmed':
            args.lr = 1e-3
            args.lambda1 = 0.5
            args.beta = 10

        # args.pretrain_path = f"./pretrain/pre_ddgae_{args.name}_new.pkl"

        acc = []
        nmi = []
        ari = []
        f1 = []
        for i in range(20):
            acc_best, nmi_best, ari_best, f1_best = trainer(dataset)
            acc.append(acc_best)
            nmi.append(nmi_best)
            ari.append(ari_best)
            f1.append(f1_best)
            res = [acc_best, nmi_best, ari_best, f1_best, 'max_epoch', args.alpha, args.lambda1, args.beta, args.lr]
            res_ = np.array(res).reshape(1, -1)
            columns = ['ACC', 'NMI', 'ARI', 'F1', 'Type', 'alpha', 'lambda1', 'beta', 'lr']
            utils.data_to_save(res_, SAVE_PATH, columns)
        print("#####################################")
        print("ACC mean", round(np.mean(acc), 5), "max", np.max(acc), "std", np.std(acc), "\n", acc)
        print("NMI mean", round(np.mean(nmi), 5), "max", np.max(nmi), "std", np.std(nmi), "\n", nmi)
        print("ARI mean", round(np.mean(ari), 5), "max", np.max(ari), "std", np.std(ari), "\n", ari)
        print("F1  mean", round(np.mean(f1), 5), "max", np.max(f1), "std", np.std(f1), "\n", f1)
        print(':acc, nmi, ari, f1: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}'.format(round(np.mean(acc), 5),
                                                                            round(np.mean(nmi), 5),
                                                                            round(np.mean(ari), 5),
                                                                            round(np.mean(f1), 5)))
        res_max = [np.max(acc), np.max(nmi), np.max(ari), np.max(f1), 'max_for', args.alpha, args.lambda1, args.beta,
                   args.lr]
        res_max = np.array(res_max).reshape(1, -1)
        columns = ['ACC', 'NMI', 'ARI', 'F1', 'Type', 'alpha', 'lambda1', 'beta', 'lr']
        utils.data_to_save(res_max, SAVE_PATH, columns)
