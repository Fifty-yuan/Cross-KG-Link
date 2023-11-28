# -*- coding:utf-8 -*-
# Anthor:fifty-yuan
# Time:2022/11/23 6:04 下午
import time

import torch
import loadData as LD
import Parameter as P
import torch.utils.data as data
from torch import nn
# import matplotlib.pyplot as plt
from gcn import GCN
import numpy as np
from tqdm import tqdm
from dataset import GcnDataset
import test_utils as tu

def train(edge_index):
    print('train start...')
    trainLoader.dataset.ng_sample()
    edge_index = edge_index.to(device)
    for epoch in range(1, P.Epoch+1):
        # 开启模型训练模式(module自带函数)
        gcnModel.train()
        All_loss = 0.0
        print('第{0}次开始：'.format(epoch))
        for idx, sample in enumerate(tqdm(trainLoader)):
            # 从sample中取值
            head, edge, tail, ng_tail, ng_rela = sample
            head = head.to(device)
            edge = edge.to(device)
            tail = tail.to(device)
            ng_tail = ng_tail.to(device)
            ng_rela = ng_rela.to(device)
            pos_score, ng_score, pos_pre, ng_pre = gcnModel(edge_index, head, edge, tail, ng_tail, ng_rela)
            # pos_pre, ng_pre = gcnModel(edge_index, head, edge, tail, ng_tail, ng_rela)
            pos_pre = pos_pre.to(torch.float32)
            ng_pre = ng_pre.to(torch.float32)
            y1 = torch.ones_like(head, dtype=torch.float32).to(device)
            y3 = torch.zeros_like(head, dtype=torch.float32).to(device)

            # 优化器
            optimizer1.zero_grad()
            # 计算预测和样本标签的损失
            loss1 = gcnLoss(ng_score, pos_score, y1)
            loss2 = tfLoss(pos_pre, y1) + tfLoss(ng_pre, y3)

            loss = loss1 + loss2
            All_loss += loss
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer1.step()

        print('第{0}次训练结束，loss1：{1}'.format(epoch, All_loss))
        if epoch % 100 == 0:
            gcnModel.saveModel(edge_index)
            torch.save(gcnModel.state_dict(), P.GcnModel_FilePath)
            evaluate()

    print('train ending...')



def evaluate():
    start = time.time()
    print('test start...')

    test_total = len(test_twoKB)

    twoKB_edge_mr = 0
    twoKB_hit1 = 0
    twoKB_hit5 = 0
    twoKB_hit10 = 0
    oneKB_edge_mr = 0
    oneKB_hit1 = 0
    oneKB_hit5 = 0
    oneKB_hit10 = 0

    # 获取训练完的节点关系embedding表示
    gcnEmbeeding = torch.load(P.GcnEmbedding_FilePath)
    node = gcnEmbeeding['node']
    edge = gcnEmbeeding['edge']

    # test the twoKB part
    for i, golden_triple in enumerate(test_twoKB):
        # test_edge（测试当前关系的概率）返回比当前头节点概率更高的关系数res（也就是排名）；以及res减去这些节点已经在训练集中出现的个数sub，即res-sub（过滤测试集的排名）
        # edge_pos, edge_pos_filter = tu.test_edge(golden_triple, trainTup, node, edge)
        edge_pos = tu.final_test(golden_triple, node, edge)
        twoKB_edge_mr += edge_pos
        if edge_pos <= 1:
            twoKB_hit1 += 1
        if edge_pos <= 5:
            twoKB_hit5 += 1
        if edge_pos <= 10:
            twoKB_hit10 += 1
    # test the oneKB part
    for i, golden_triple in enumerate(test_oneKB):
        # test_edge（测试当前关系的概率）返回比当前头节点概率更高的关系数res（也就是排名）；以及res减去这些节点已经在训练集中出现的个数sub，即res-sub（过滤测试集的排名）
        edge_pos = tu.final_test(golden_triple, node, edge)
        if i % 500 == 0:
            print(golden_triple[:3], end=': ')
            print('edge_pos=' + str(edge_pos), end=', ')
            print('test ---' + str(i) + '--- triple')
            print(i, end="\r")
            # print('edge_pos_filter=' + str(edge_pos_filter), end=', ')
            # 所有排名相加，过滤排名相加 最后再算平均
        oneKB_edge_mr += edge_pos
        # edge_mr_filter += edge_pos_filter
        if edge_pos == 1:
            oneKB_hit1 += 1
        if edge_pos <= 5:
            oneKB_hit5 += 1
        if edge_pos <= 10:
            oneKB_hit10 += 1
    # 算平均
    twoKB_edge_mr /= test_total
    twoKB_hit1 /= test_total
    twoKB_hit5 /= test_total
    twoKB_hit10 /= test_total
    oneKB_edge_mr /= test_total
    oneKB_hit1 /= test_total
    oneKB_hit5 /= test_total
    oneKB_hit10 /= test_total

    # total
    total_edge_mr = (twoKB_edge_mr + oneKB_edge_mr) / 2
    total_hit1 = (twoKB_hit1 + oneKB_hit1) / 2
    total_hit5 = (twoKB_hit5 + oneKB_hit5) / 2
    total_hit10 = (twoKB_hit10 + oneKB_hit10) / 2

    # \t代表空格
    print('\t\t\tmean_rank\t\t\t')
    # %代表格式化输出 .3f控制小数点后三位
    print('total_edge(raw)\t\t\t%.3f\t\t\t' % total_edge_mr)
    print('total_hit@1\t\t\t%.3f\t\t\t' % total_hit1)
    print('total_hit@5\t\t\t%.3f\t\t\t' % total_hit5)
    print('total_hit@10\t\t\t%.3f\t\t\t' % total_hit10)
    print('twoKB_edge_mr(raw)\t\t\t%.3f\t\t\t' % twoKB_edge_mr)
    print('twoKB_hit@1\t\t\t%.3f\t\t\t' % twoKB_hit1)
    print('twoKB_hit@5\t\t\t%.3f\t\t\t' % twoKB_hit5)
    print('twoKB_hit@10\t\t\t%.3f\t\t\t' % twoKB_hit10)
    print('oneKB_edge_mr(raw)\t\t\t%.3f\t\t\t' % oneKB_edge_mr)
    print('oneKB_hit@1\t\t\t%.3f\t\t\t' % oneKB_hit1)
    print('oneKB_hit@5\t\t\t%.3f\t\t\t' % oneKB_hit5)
    print('oneKB_hit@10\t\t\t%.3f\t\t\t' % oneKB_hit10)
    end = time.time()
    print("ues time %s second" % (end-start))


if __name__ == '__main__':

    LD.set_random_seed(P.seed)

    edge_index, train_data, test_data, trainTup = LD.Load_data()
    # divide test_data into two part: triples from two kb and triples from one kb, this two part have the same num. dfb1 and wk3l 20000, medicine 1000
    for i in range(len(test_data)):
        if len(test_data[i]) != 3:
            test_data[i] = test_data[i][:3]
    filter_tri = dict()
    for i in range(len(trainTup)):
        filter_tri[trainTup[i]] = i
    for i in range(len(test_data)):
        t = (test_data[i][0], test_data[i][1], test_data[i][2])
        filter_tri[t] = i
    test_twoKB = test_data[:20000]
    test_oneKB = test_data[20000:]


    trainDataSet = GcnDataset(train_data, is_training=True)
    trainLoader = data.DataLoader(trainDataSet, batch_size=P.Batch_Size, shuffle=True)

    device = torch.device(P.Gpu if torch.cuda.is_available() else 'cpu')

    gcnModel = GCN().to(device)


    gcnLoss = nn.MarginRankingLoss(margin=P.Margin, reduction='mean')
    tfLoss = nn.MSELoss(reduction='mean')

    optimizer1 = torch.optim.Adam(gcnModel.parameters(), lr=P.Learning_Rate)
    train(edge_index)