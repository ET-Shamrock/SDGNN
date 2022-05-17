# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: huangjunjie
@file: sdgnn.py
@time: 2019/12/10
"""

import os
import sys
import time
import math
import random
import subprocess
import logging
import argparse
import pandas as pd

from collections import defaultdict

import numpy as np
import scipy
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from zmq import device


from common import DATASET_NUM_DIC
#
from fea_extra import FeaExtra
from logistic_function import logistic_embedding



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--devices', type=str, default='cpu', help='Devices')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dataset', default='bitcoin_otc', help='Dataset')
parser.add_argument('--dim', type=int, default=20, help='Embedding dimension')
parser.add_argument('--fea_dim', type=int, default=30, help='Feature embedding dimension')
parser.add_argument('--batch_size', type=int, default=400, help='Batch size')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout k')
parser.add_argument('--k', default=1, help='Folder k')
parser.add_argument('--agg', default='mean', choices=['mean', 'attantion'], help='Aggregator choose')

args = parser.parse_args()

OUTPUT_DIR = f'./embeddings/sdgnn-{args.agg}'
if not os.path.exists('embeddings'):
    os.mkdir('embeddings')

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
        
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

NEG_LOSS_RATIO = 1
INTERVAL_PRINT = 2

NUM_NODE = 5000
WEIGHT_DECAY = args.weight_decay
NODE_FEAT_SIZE = args.fea_dim
EMBEDDING_SIZE1 = args.dim
DEVICES = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICES)
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
DROUPOUT = args.dropout
K = args.k


class Encoder(nn.Module):
    """
    Encode features to embeddings
    """

    def __init__(self, features, feature_dim, embed_dim, adj_lists, aggs, W):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggs = aggs
        self.W = W

        self.embed_dim = embed_dim
        # add_module 可以给nn加层
        # 加了四个 aggregator
        for i, agg in enumerate(self.aggs):
            self.add_module('agg_{}'.format(i), agg)
            self.aggs[i] = agg.to(DEVICES)

        def init_weights(m):
            if type(m) == nn.Linear:
                # 权重 normalize
                torch.nn.init.kaiming_normal_(m.weight)
                # 计算s输入层的个数
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                # bias uniform
                nn.init.uniform_(m.bias, -bound, bound)

        # 网络
        self.nonlinear_layer = nn.Sequential(
                # len(adj_lists)是有几种边的关系，+1是因为要把自身结点的编码也concatenate进去
                nn.Linear((len(adj_lists) + 1) * feature_dim, feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, embed_dim)
        )

        self.nonlinear_layer.apply(init_weights)


    def forward(self, nodes):
        """
        Generates embeddings for nodes.
        """

        if not isinstance(nodes, list) and nodes.is_cuda:
            nodes = nodes.data.cpu().numpy().tolist()

        neigh_feats = [agg(nodes, adj, self.W, ind) for adj, agg, ind in zip(self.adj_lists, self.aggs, range(len(self.adj_lists)))]
        self_feats = self.features(torch.LongTensor(nodes).to(DEVICES))
        combined = torch.cat([self_feats] + neigh_feats, 1)
        combined = self.nonlinear_layer(combined)
        return combined


class AttentionAggregator(nn.Module):
    def __init__(self, features, in_dim, out_dim, node_num,  dropout_rate=DROUPOUT, slope_ratio=0.1):
        super(AttentionAggregator, self).__init__()

        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.slope_ratio = slope_ratio
        self.a = nn.Parameter(torch.FloatTensor(out_dim * 2, 1))
        nn.init.kaiming_normal_(self.a.data)

        self.out_linear_layer = nn.Linear(self.in_dim, self.out_dim)
        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)


    def forward(self, nodes, adj, ind):
        """
        nodes --- list of nodes in a batch
        adj --- sp.csr_matrix
        """
        node_pku = np.array(nodes)
        edges = np.array(adj[nodes, :].nonzero()).T
        edges[:, 0] = node_pku[edges[:, 0]]

        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))

        batch_node_num = len(unique_nodes_list)
        # this dict can map new i to originial node id
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)

        edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]

        n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)
        new_embeddings = self.out_linear_layer(self.features(n2))

        original_node_edge = np.array([self.unique_nodes_dict[nodes], self.unique_nodes_dict[nodes]]).T
        edges = np.vstack((edges, original_node_edge))

        edges = torch.LongTensor(edges).to(DEVICES)

        edge_h_2 = torch.cat((new_embeddings[edges[:, 0], :], new_embeddings[edges[:, 1], :]), dim=1)

        edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), self.slope_ratio))
        indices = edges
        
        matrix = torch.sparse_coo_tensor(indices.t(), edges_h[:, 0], \
                                         torch.Size([batch_node_num, batch_node_num]), device=DEVICES)
        row_sum = torch.sparse.mm(matrix, torch.ones(size=(batch_node_num, 1)).to(DEVICES))

        results = torch.sparse.mm(matrix, new_embeddings)

        output_emb = results.div(row_sum)

        return output_emb[self.unique_nodes_dict[nodes]]

class MeanAggregator(nn.Module):
    def __init__(self, features, in_dim, out_dim, node_num):
        super(MeanAggregator, self).__init__()

        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_linear_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.Tanh(),
            nn.Linear(self.out_dim, self.out_dim)
        )

        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)

    def forward(self, nodes, adj, W, ind):
        """
        :param nodes:
        :param adj:
        :return:
        """
        mask = [1, 1, 0, 0]
        # batch训练的结点编号
        node_tmp = np.array(nodes)
        # 获得非零元素的索引，也就是获得邻居的编号（因为此时的adj是一个邻接矩阵）
        edges = np.array(adj[nodes, :].nonzero()).T
        # 把下标转成结点编号。因为edges的第0列是 [0, len(nodes)-1]的范围，但是我们需要的是结点编号，所以需要转换一下
        edges[:, 0] = node_tmp[edges[:, 0]]
        # 处理完的edges是邻接表了，每一行都是一个二元组，表示 (i, j)有一条边

        #--------------离散化的过程--------------
        #所有节点去重
        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))
        #去重后的结点数目
        batch_node_num = len(unique_nodes_list)
        #对去重后的结点编号
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)
        #---------------------------------------
        W_temp = []
        for i in range(edges.shape[0]):
            u, v = edges[i][0], edges[i][1]
            uu = self.unique_nodes_dict[u]
            vv = self.unique_nodes_dict[v]
            W_temp.append(W[u][v])
            edges[i][0] = uu
            edges[i][1] = vv

        # print(W_temp)

        ## transform 2 new axis，转换成 [0, num_nodes - 1]，上面循环可以代替这个功能
        # edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        # edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]

        n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)
        # 对本次batch涉及到的所有nodes进行embedding，相当于计算z_l
        new_embeddings = self.out_linear_layer(self.features(n2))
        edges = torch.LongTensor(edges).to(DEVICES)
        W_temp = torch.FloatTensor(W_temp).to(DEVICES)

        values = torch.where(edges[:, 0] == edges[:, 1], torch.FloatTensor([mask[ind]]).to(DEVICES), torch.FloatTensor([1]).to(DEVICES))
        # print(values)
        # values = torch.ones(edges.shape[0]).to(DEVICES)

        # 构建一个 Coordinate矩阵，edges.t()是求转置
        # matrix_neighbor是邻居的“邻居个数”权重， 也就是各个embedding的权重矩阵（0-1矩阵）
        # 当有自环的时候，在前两种边的时候考虑进去了，那么后两种边就不考虑了，所以mask = [1, 1, 0, 0]
        matrix_coo = torch.sparse_coo_tensor(edges.t(), values, torch.Size([batch_node_num, batch_node_num]), device=DEVICES)
        
        # 每行求和，得到出度；转置并纵向复制
        node_degree = torch.spmm(matrix_coo, torch.ones(size=(batch_node_num, 1)).to(DEVICES)).t().repeat(batch_node_num, 1)

        # 获得指定位置的出度
        index = edges[:, 0] * batch_node_num + edges[:, 1]
        # take是把matrix拉平，取值
        temp = torch.take(node_degree, index)
        # print(temp.shape)
        # print(temp)

        # 不能直接 sparse * dense，好麻烦
        # matrix_neighbor = matrix_coo.mul(node_degree)

        # neig[i][j] 表示的是j的出度
        matrix_neighbor = torch.sparse_coo_tensor(edges.t(), temp, torch.Size([batch_node_num, batch_node_num]), device=DEVICES)

        # matrix_weight 是考虑边权了
        matrix_weight = torch.sparse_coo_tensor(edges.t(), W_temp, torch.Size([batch_node_num, batch_node_num]), device=DEVICES)

        # 两种加权放到一起（点乘）
        matrix = matrix_neighbor.mul(matrix_weight)

        # spmm是矩阵乘法，每行求和，用于后面求平均
        row_sum = torch.spmm(matrix, torch.ones(size=(batch_node_num, 1)).to(DEVICES))

        # 把和为0的行变成1，因为后面有除法运算，除数不能为0
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(DEVICES), row_sum)

        results = torch.spmm(matrix, new_embeddings)
        # 求平均
        output_emb = results.div(row_sum)

        return output_emb[self.unique_nodes_dict[nodes]]



class SDGNN(nn.Module):

    def __init__(self, enc):
        super(SDGNN, self).__init__()
        self.enc = enc
        # 计算score的
        self.score_function1 = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE1, 1),
            nn.Sigmoid()
        )
        # 计算score的
        self.score_function2 = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE1, 1),
            nn.Sigmoid()
        )
        # 计算L_triangel的
        self.fc = nn.Linear(EMBEDDING_SIZE1 * 2, 1)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        return embeds

    def criterion(self, nodes, pos_neighbors, neg_neighbors, adj_lists1_1, adj_lists2_1, weight_dict):
        pos_neighbors_list = [set.union(pos_neighbors[i]) for i in nodes]
        neg_neighbors_list = [set.union(neg_neighbors[i]) for i in nodes]
        unique_nodes_list = list(set.union(*pos_neighbors_list).union(*neg_neighbors_list).union(nodes))
        # 离散化映射，node_id -> index
        unique_nodes_dict = {n: i for i, n in enumerate(unique_nodes_list)}
        nodes_embs = self.enc(unique_nodes_list)

        loss_total = 0
        for index, node in enumerate(nodes):
            # 当前结点的embedding
            z1 = nodes_embs[unique_nodes_dict[node], :]

            pos_neigs = list([unique_nodes_dict[i] for i in pos_neighbors[node]])
            neg_neigs = list([unique_nodes_dict[i] for i in neg_neighbors[node]])
            pos_num = len(pos_neigs)
            neg_num = len(neg_neigs)

            # 正出边的邻居
            sta_pos_neighs = list([unique_nodes_dict[i] for i in adj_lists1_1[node]])
            # 负出边的邻居
            sta_neg_neighs = list([unique_nodes_dict[i] for i in adj_lists2_1[node]])

            # 满足条件的三角形个数
            pos_neigs_weight = torch.FloatTensor([weight_dict[node][i] for i in adj_lists1_1[node]]).to(DEVICES)
            neg_neigs_weight = torch.FloatTensor([weight_dict[node][i] for i in adj_lists2_1[node]]).to(DEVICES)

            if pos_num > 0:
                # 正邻居的embedding
                pos_neig_embs = nodes_embs[pos_neigs, :]
                # 计算 L_sign
                # binary_cross_entropy_with_logits 是计算交叉熵的，但是不要求输入是在[0, 1]之间的， 函数内置sigmoid函数进行转换
                # 这是正邻居，所以sign都是正边，因此是torch.ones()
                loss_pku = F.binary_cross_entropy_with_logits(torch.einsum("nj,j->n", [pos_neig_embs, z1]),torch.ones(pos_num).to(DEVICES))

                # 计算 L_triangle和 L_direction
                if len(sta_pos_neighs) > 0:
                    sta_pos_neig_embs = nodes_embs[sta_pos_neighs, :]
                    # 纵向复制，使得行数和neighbors一样
                    z11 = z1.repeat(len(sta_pos_neighs), 1)
                    # 邻居的embedding和自身的cat（纵向cat），因此第2维的size是 2*embeddingsize, 这也是为什么self.fc的维度设置成2倍了
                    rs = self.fc(torch.cat([z11, sta_pos_neig_embs], 1)).squeeze(-1)
                    # 加权交叉熵损失函数，weight设置为对于每一对(u, v)满足条件的三角形个数
                    loss_pku += F.binary_cross_entropy_with_logits(rs, torch.ones(len(sta_pos_neighs)).to(DEVICES), \
                                                                   weight=pos_neigs_weight
                                                                   )
                    s1 = self.score_function1(z1).repeat(len(sta_pos_neighs), 1)
                    s2 = self.score_function2(sta_pos_neig_embs)

                    q = torch.where((s1 - s2) > -0.5, torch.Tensor([-0.5]).repeat(s1.shape).to(DEVICES), s1 - s2)
                    tmp = (q - (s1 - s2))
                    # 对应点相乘再求和，这也就是平方和了
                    loss_pku += 5 * torch.einsum("ij,ij->", [tmp, tmp])

                loss_total += loss_pku

            # 同上
            if neg_num > 0:
                neg_neig_embs = nodes_embs[neg_neigs, :]
                loss_pku = F.binary_cross_entropy_with_logits(torch.einsum("nj,j->n", [neg_neig_embs, z1]),
                                                              torch.zeros(neg_num).to(DEVICES))
                if len(sta_neg_neighs) > 0:
                    sta_neg_neig_embs = nodes_embs[sta_neg_neighs, :]

                    z12 = z1.repeat(len(sta_neg_neighs), 1)
                    rs = self.fc(torch.cat([z12, sta_neg_neig_embs], 1)).squeeze(-1)

                    # 负边得到的label是0
                    loss_pku += F.binary_cross_entropy_with_logits(rs, torch.zeros(len(sta_neg_neighs)).to(DEVICES), \
                                                                   weight=neg_neigs_weight)

                    s1 = self.score_function1(z1).repeat(len(sta_neg_neighs), 1)
                    s2 = self.score_function2(sta_neg_neig_embs)

                    q = torch.where(s1 - s2 > 0.5, s1 - s2, torch.Tensor([0.5]).repeat(s1.shape).to(DEVICES))

                    tmp = (q - (s1 - s2))
                    loss_pku += 5 * torch.einsum("ij,ij->", [tmp, tmp])

                loss_total += loss_pku

        return loss_total


def load_data2(filename=''):
    # 相当于 map<int, set> 实现邻接表
    adj_lists1 = defaultdict(set)
    adj_lists1_1 = defaultdict(set)
    adj_lists1_2 = defaultdict(set)
    adj_lists2 = defaultdict(set)
    adj_lists2_1 = defaultdict(set)
    adj_lists2_2 = defaultdict(set)
    adj_lists3 = defaultdict(set)
    W = defaultdict(dict)
    # W = np.zeros((NUM_NODE, NUM_NODE), dtype=np.float64)
    
    df = pd.read_csv(filename)
    edges = df.values
    # print(edges[:10, :])

    for i, edge in enumerate(edges):
        person1 = int(edge[0])
        person2 = int(edge[1])
        weight = edge[2]
        W[person1][person2] = W[person2][person1] = abs(weight)
        adj_lists3[person2].add(person1)
        adj_lists3[person1].add(person2)

        # 正邻居
        if weight > 0:
            # 结点之间的关系
            adj_lists1[person1].add(person2)
            adj_lists1[person2].add(person1)

            # 正出边
            adj_lists1_1[person1].add(person2)
            # 正入边
            adj_lists1_2[person2].add(person1)

        # 负邻居
        elif weight < 0:
            adj_lists2[person1].add(person2)
            adj_lists2[person2].add(person1)
            # 负出边
            adj_lists2_1[person1].add(person2)
            # 负入边
            adj_lists2_2[person2].add(person1)
    # print(W)
    return adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2, adj_lists3, W



def read_emb(num_nodes, fpath):
    dim = 0
    embeddings = 0
    with open(fpath) as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                dim = int(line.split()[1])
                embeddings = np.random.rand(num_nodes, dim)
            else:
                line_l = line.split()
                node = line_l[0]
                emb = [float(j) for j in line_l[1:]]
                assert len(emb) == dim
                embeddings[int(node)] = np.array(emb)
    return embeddings


def run():
    num_nodes = NUM_NODE
    filename = '../data/Slashdot_sub_0.01.csv'
    
    # adj_lists1, adj_lists2, adj_lists3 = load_data(k, dataset)
    # filename = './experiment-data/{}-train-{}.edgelist'.format(dataset, k)
    
    adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2, adj_lists3, W = load_data2(filename)
    # print(k, dataset, 'data load!')

    # 对每个结点进行 embedding，dim = NODE_FEAT_SIZE
    # features是embedding类的一个instance，调用的时候是 features(indices)
    features = nn.Embedding(num_nodes, NODE_FEAT_SIZE)
    features.weight.requires_grad = True
    features = features.to(DEVICES)

    # 正出边、正入边、负出边、负入边
    adj_lists = [adj_lists1_1, adj_lists1_2,  adj_lists2_1, adj_lists2_2]

    
    weight_dict = defaultdict(dict)
    fea_model = FeaExtra(filename)

    #枚举正出边
    for i in adj_lists1_1:
        for j in adj_lists1_1[i]:
            v_list1 = fea_model.feature_part2(i, j)
            # 权重为1的是可以满足三角形要求的情况（i->j为正边权）
            mask = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]
            counts1 = np.dot(v_list1, mask)
            weight_dict[i][j] = counts1

    #枚举负出边（同上）
    for i in adj_lists2_1:
        for j in adj_lists2_1[i]:
            v_list1 = fea_model.feature_part2(i, j)
            mask = [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]

            counts1 = np.dot(v_list1, mask)
            weight_dict[i][j] = counts1

    adj_lists = adj_lists


    print(len(adj_lists), 'motifs')

    # 邻接表转邻接矩阵
    def func(adj_list):
        edges = []
        for a in adj_list:
            for b in adj_list[a]:
                edges.append((a, b))
        edges = np.array(edges)
        # 构造稀疏矩阵
        adj = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
        return adj

    if args.agg == 'mean':
        aggregator = MeanAggregator
    else:
        aggregator = AttentionAggregator

    adj_lists = list(map(func, adj_lists))
    # 四种关系，构建四个 aggregator，每个adj是一个邻接矩阵
    aggs = [aggregator(features, NODE_FEAT_SIZE, NODE_FEAT_SIZE, num_nodes) for adj in adj_lists]

    # 第一层Encoder
    enc1 = Encoder(features, NODE_FEAT_SIZE, EMBEDDING_SIZE1, adj_lists, aggs, W)
    enc1 = enc1.to(DEVICES)

    # 第二层Encoder
    aggs2 = [aggregator(lambda n: enc1(n), EMBEDDING_SIZE1, EMBEDDING_SIZE1, num_nodes) for _ in adj_lists]
    enc2 = Encoder(lambda n: enc1(n), EMBEDDING_SIZE1, EMBEDDING_SIZE1, adj_lists, aggs2, W)

    model = SDGNN(enc2)
    model = model.to(DEVICES)

    print(model.train())
    # 优化器，只优化需要梯度信息的parameter
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,list(model.parameters()) + list(enc1.parameters()) \
                                        + list(features.parameters())),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY
                                 )

    for epoch in range(EPOCHS + 2):
        total_loss = []
        # 保存embedding的结果
        if epoch % INTERVAL_PRINT == 0:
            model.eval()
            all_embedding = np.zeros((NUM_NODE, EMBEDDING_SIZE1))
            for i in range(0, NUM_NODE, BATCH_SIZE):
                begin_index = i
                end_index = i + BATCH_SIZE if i + BATCH_SIZE < NUM_NODE else NUM_NODE
                values = np.arange(begin_index, end_index)
                embed = model.forward(values.tolist())
                embed = embed.data.cpu().numpy()
                all_embedding[begin_index: end_index] = embed

            fpath = os.path.join(OUTPUT_DIR, 'embedding-{}.npy'.format(str(epoch)))
            np.save(fpath, all_embedding)
            # pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = logistic_embedding(k=k, dataset=dataset,epoch=epoch,dirname=OUTPUT_DIR)
            
            model.train()

        time1 = time.time()
        # 随机打乱结点数据
        nodes_pku = np.random.permutation(NUM_NODE).tolist()
        for batch in range(NUM_NODE // BATCH_SIZE):
            optimizer.zero_grad()
            b_index = batch * BATCH_SIZE
            e_index = (batch + 1) * BATCH_SIZE
            nodes = nodes_pku[b_index:e_index]

            # 正邻居、负邻居、正出边、负出边
            loss = model.criterion(
                nodes, adj_lists1, adj_lists2, adj_lists1_1, adj_lists2_1, weight_dict
            )

            # 取loss的data，防止内存爆炸
            total_loss.append(loss.data.cpu().numpy())

            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch}, loss: {np.mean(total_loss)}, time: {time.time()-time1}')

def main():
    print('NUM_NODE', NUM_NODE)
    print('WEIGHT_DECAY', WEIGHT_DECAY)
    print('NODE_FEAT_SIZE', NODE_FEAT_SIZE)
    print('EMBEDDING_SIZE1', EMBEDDING_SIZE1)
    print('LEARNING_RATE', LEARNING_RATE)
    print('BATCH_SIZE', BATCH_SIZE)
    print('EPOCHS', EPOCHS)
    print('DROUPOUT', DROUPOUT)
    run()


if __name__ == "__main__":
    main()

