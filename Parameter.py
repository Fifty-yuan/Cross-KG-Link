# -*- coding:utf-8 -*-
# Anthor:fifty-yuan
# Time:2022/11/22 4:25 下午

# python版本3.8
# pytorch版本1.11.1
# 服务器CUDA版本10.2

# embedding的维度
Emb_Dimension = 256

# 每次传入多少条数据
Batch_Size = 256

# 训练轮次 TIdfb1:300 wk3l:100
Epoch = 200

# 学习速率
Learning_Rate = 0.00001

# dropout率
Drop_Out = 0

# 损失函数的margin
Margin = 0.3

# norm
Norm = 1

# 负采样倍数，即负采样是正样本的几倍
Negative_Sample = 1

# 训练集占总数据的百分之几
Train_Num = 0.8

# 是否需要偏置
Bias = False

# 矩阵是否稀疏
TensorSparse = True

# 是否小样本测试
LessTestSample = False

# 小样本数量
LessTestNum = 3000

# 随即种子seed
seed = 2022

# subject总个数 家用KB：486 说明书KB：734 DFB1KB:24902 wk3l:64539 medicine:8968
Subject_Num = 64539

# object总个数 家用KB：5017 说明书KB：5570 DFB1KB:24902
Object_Num = 64539

# 关系种类数量 家用KB：9 说明书KB：11 DFB1：1345 wk3l:458 medicine:15
Relation_Num = 458

# 输入数据地址 dataset为家用KB 2为说明书KB.
Data_Path= '/home/lbr/wsyshell/TFwithGCN/data/wk3lmethod2-30%-done.pt'

# 保留gcn训练后节点和关系embedding的地址
GcnEmbedding_FilePath = '/home/lbr/wsyshell/TFwithGCN/modelParameter/gcnEmbedding.pth'

# 模型保存地址 服务器地址
GcnModel_FilePath = '/home/lbr/wsyshell/TFwithGCN/modelParameter/model_file.pth'

# GPU选择
Gpu = 'cuda:3'

# 若要选用多个GPU
Gpu_id = [1, 2, 3]
