# -*- coding:utf-8 -*-
# Anthor:fifty-yuan
# Time:2022/12/4 12:25 上午
import random
from torch.utils.data import Dataset
import Parameter as P


class GcnDataset(Dataset):
    # 初始化函数
    def __init__(self, triples, is_training = True) -> None:
        print("=" * 10, "Creating Dataset", "=" * 10)
        self.triples = triples
        self.objectSet = {i for i in range(P.Object_Num)}
        self.is_training = is_training
        self.relationSet = {i for i in range(P.Relation_Num)}
        self.n_triple = P.Negative_Sample * len(triples)
        # if self.is_training:
        #     self.n_triple = (1 + P.Negative_Sample) * len(triples)
        # else:
        #     self.n_triple = len(triples)

    # 返回数据的长度
    def __len__(self):
        return self.n_triple

    # 根据index返回数据
    def __getitem__(self, index):
        # 如果是训练集的话就返回正负样本
        if self.is_training:
            sub_id, rela_id, ob_id, ngob_id, ngrela_id = self.po_ng_set[index]
            sample = sub_id, rela_id, ob_id, ngob_id, ngrela_id
        # 测试集无负样本
        else:
            sub_id, rela_id, ob_id, ngob_id = self.po_ng_set[index]
            sample = sub_id, rela_id, ob_id, ngob_id
        return sample

    # 生成负样本
    def ng_sample(self):
        self.po_ng_set = []
        for triple in self.triples:
            h, r, tail = triple[0], triple[1], triple[2]
            for t in range(P.Negative_Sample):
                # np_object是负样本的尾节点集合，从中抽取的节点都可以作为该三元组的负样本，triple[3]放的是当前h，r对应的所有t集合
                np_object = self.objectSet - set(triple[3])
                # j为负样本的尾节点，注意这里虽然sample出来1个数，但是该函数返回的是一个列表
                j = random.sample(np_object, 1)
                # 在该集合中加入j，保证下次负采样的时候不会采到同样的j
                ng_r = random.sample(self.relationSet, 1)
                triple[3].append(j[0])
                # 正(hrt)负(hrj)样本生成
                poNgSam = tuple([h, r, tail, j[0], ng_r[0]])
                self.po_ng_set.append(poNgSam)

    def gcn_ng_sample(self):
        self.po_ng_set = []
        for triple in self.triples:
            self.relationSet = {i for i in range(P.Relation_Num)}
            h, r, tail = triple[0], triple[1], triple[2]
            for t in range(P.Negative_Sample):
                # np_object是负样本的尾节点集合，从中抽取的节点都可以作为该三元组的负样本，triple[3]放的是当前h，r对应的所有t集合
                np_object = self.objectSet
                # j为负样本的尾节点，注意这里虽然sample出来1个数，但是该函数返回的是一个列表
                j = random.sample(np_object, 1)
                if(j[0] == tail):
                    j = random.sample(np_object, 1)
                # 在该集合中加入j，保证下次负采样的时候不会采到同样的j
                ng_r = random.sample(self.relationSet, 1)
                # triple[3].append(j[0])
                # 正(hrt)负(hrj)样本生成
                poNgSam = tuple([h, r, tail, j[0], ng_r[0]])
                self.po_ng_set.append(poNgSam)
