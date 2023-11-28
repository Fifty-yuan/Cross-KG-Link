# -*- coding:utf-8 -*-
# Anthor:fifty-yuan
# Time:2022/11/22 4:25 下午
import Parameter as P
import torch
import random

def Load_dirdata():

    # 根据路径读取dataset
    dataset = torch.load(P.Data_Path)
    triples = dataset['pos_tri']
    # triples是所有数据（已经打乱），接下来分出训练集和测试集，trainNum为测试集数量
    trainNum = int(len(triples) * P.Train_Num)
    train_data = triples[:trainNum-1]
    test_data = triples[trainNum-1:]
    return dataset, train_data, test_data


def Load_data():
    # 根据路径读取dataset
    dataset = torch.load(P.Data_Path)
    train_data = dataset['pos_tri']
    test_data = dataset['test_tri']
    if P.LessTestSample:
        random.seed(P.seed)
        test_data = random.sample(test_data, P.LessTestNum)
    # 构造edge_index
    head, tail = [], []
    trainTup = dataset['traintup']
    for triple in train_data:
        head.append(triple[0])
        tail.append(triple[2])
    edge_index = [head, tail]
    edge_index = torch.Tensor(edge_index).to(int)
    return edge_index, train_data, test_data, trainTup

def Load_gcndata():
    # 根据路径读取dataset
    dataset = torch.load(P.Data_Path)
    train_data = dataset['pos_tri']
    test_data = dataset['test_tri']
    # 构造edge_index
    # max_node, max_rel = 0, 0
    head, tail, rel = [], [], []
    for triple in train_data:
        # if(triple[0] > max_node):
        #     max_node = triple[0]
        # if(triple[1] > max_rel):
        #     max_rel = triple[1]
        head.append(triple[0])
        head.append(triple[2])
        rel.append(triple[1])
        rel.append(triple[1])
        tail.append(triple[2])
        tail.append(triple[0])
    edge_index = [head, tail, rel]
    edge_index = torch.Tensor(edge_index).to(int)
    return edge_index, train_data, test_data


def set_random_seed(seed):
    '''
        固定模型随机种子
        input: P.seed
        output: none
    '''
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
