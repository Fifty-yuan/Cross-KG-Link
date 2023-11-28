# -*- coding:utf-8 -*-
# Anthor:fifty-yuan
# Time:2022/12/6 3:42 上午
import Parameter as P
import numpy as np
import torch
from gcn import GCN

device = torch.device(P.Gpu if torch.cuda.is_available() else 'cpu')


def test_head(golden_triple, entity_emb, relation_emb):
    """

    :param golden_triple:
    :param train_set:
    :param entity_emb:
    :param relation_emb:
    :return:
    """
    # head_batch:构造出来的三元组，数量为节点的总数量（每一个节点都作为头节点预测可能性），头节点是0到n-1，关系和尾节点为当前golden_triple的关系和尾节点
    head_batch = get_head_batch(golden_triple)
    # value返回head_batch里面所有三元组的预测分数（预测分数越小概率越高）
    value = predict(head_batch, entity_emb, relation_emb)
    # 取出当前三元组真正的头节点golden_triple[0]所对应的分数
    _, indices = torch.sort(value, descending=False)
    rank = int((indices == golden_triple[0]).nonzero())

    return rank + 1


def test_edge(golden_triple, filter_tri, entity_emb, relation_emb):
    """

    :param golden_triple:
    :param train_set:
    :param entity_emb:
    :param relation_emb:
    :return:
    """
    edge_batch = get_edge_batch(golden_triple)


    # value返回edge_batch里面所有三元组的预测分数（预测分数越小概率越高）
    value = predict(edge_batch, entity_emb, relation_emb)
    # 取出当前三元组真正关系golden_triple[1]所对应的分数
    # golden_value = value[0]
    _, indices = torch.sort(value, descending=False)
    rank = int((indices == golden_triple[1]).nonzero())

    return rank+1

def final_test(golden_triple, entity_emb, relation_emb):

    edge_batch = get_edge_batch(golden_triple)
    value = predict(edge_batch, entity_emb, relation_emb)
    # device = torch.device(P.Gpu if torch.cuda.is_available() else 'cpu')
    # edge_batch = torch.tensor(edge_batch, dtype=torch.long).to(device)
    # model = GCN().to(device)
    # model.load_state_dict(torch.load(P.GcnModel_FilePath))
    #
    # pre = model.nnfpmc(edge_batch[:, 0], edge_batch[:, 1], edge_batch[:, 2], entity_emb)
    # pre = 1 - pre
    #
    # final_score = pre + value
    # edge_batch = edge_batch.to(device)
    # final_score = model(edge_index, edge_batch[:, 0], edge_batch[:, 1], edge_batch[:, 2])
    _, indices = torch.sort(value, descending=False)
    rank = int((indices == golden_triple[1]).nonzero())

    return rank + 1







def test_tail(golden_triple, train_set, entity_emb, relation_emb):
    tail_batch = get_tail_batch(golden_triple)
    value = predict(tail_batch, entity_emb, relation_emb)
    golden_value = value[golden_triple[2]]
    # li = np.argsort(value)
    res = 1
    sub = 0
    for pos, val in enumerate(value):
        if val < golden_value:
            res += 1
            if (golden_triple[0], golden_triple[1], pos) in train_set:
                sub += 1

    return res, res - sub





def get_head_batch(golden_triple):
    """
    :param golden_triple: 当前测试三元组
    :return: 构造出来的三元组，数量为节点的总数量n（每一个节点都作为头节点预测可能性），头节点是0到n-1，关系和尾节点为当前golden_triple的关系和尾节点
    """
    head_batch = np.zeros((P.Subject_Num, 3), dtype=np.int32)
    head_batch[:, 0] = np.array(list(range(P.Subject_Num)))
    head_batch[:, 1] = np.array([golden_triple[1]] * P.Subject_Num)
    head_batch[:, 2] = np.array([golden_triple[2]] * P.Subject_Num)

    return head_batch


def get_edge_batch(golden_triple):
    """
    :param golden_triple: 当前测试三元组
    :return:
    """
    edge_batch = np.zeros((P.Relation_Num, 3), dtype=np.int32)
    edge_batch[:, 0] = np.array([golden_triple[0]] * P.Relation_Num)
    edge_batch[:, 1] = np.array(list(range(P.Relation_Num)))
    edge_batch[:, 2] = np.array([golden_triple[2]] * P.Relation_Num)

    return edge_batch


def get_tail_batch(golden_triple):
    # 功能同上
    tail_batch = np.zeros((P.Subject_Num, 3), dtype=np.int32)
    tail_batch[:, 0] = np.array([golden_triple[0]] * P.Subject_Num)
    tail_batch[:, 1] = np.array([golden_triple[1]] * P.Subject_Num)
    tail_batch[:, 2] = np.array(list(range(P.Subject_Num)))

    return tail_batch


def predict(head_batch, entity_emb, relation_emb):
    """
    :param head_batch: 构造出来的三元组，数量为节点的总数量n（每一个节点都作为头节点预测可能性），头节点是0到n-1，关系和尾节点为当前golden_triple的关系和尾节点
    :param entity_emb: 训练完的节点的emb
    :param relation_emb: 训练完的关系的emb
    :return: 当前三元组中关系和尾节点所预测的所有头节点的分数（分数越小概率越高）
    """
    pos_hs = head_batch[:, 0]
    pos_rs = head_batch[:, 1]
    pos_ts = head_batch[:, 2]
    # 分别取出头节点，关系，尾节点。注意 这里所有关系和尾节点都是一样的

    # 本来的代码为pos_rs = torch.IntTensor(pos_rs).cuda()
    pos_hs = torch.IntTensor(pos_hs).to(device)
    pos_rs = torch.IntTensor(pos_rs).to(device)
    pos_ts = torch.IntTensor(pos_ts).to(device)

    # 将hrt对应的id转化成embedding
    h = entity_emb[pos_hs.type(torch.long)].to(device)
    r = relation_emb[pos_rs.type(torch.long)].to(device)
    t = entity_emb[pos_ts.type(torch.long)].to(device)

    # 计算所有头节点的分数
    # p_score = torch.norm(h + r - t, p=P.Norm, dim=1).detach().cpu().numpy().tolist()
    p_score = torch.norm(h + r - t, p=P.Norm, dim=1)
    return p_score

