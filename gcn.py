# -*- coding:utf-8 -*-
# Anthor:fifty-yuan
# Time:2022/11/15 5:21 下午
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
import Parameter as P

device = torch.device(P.Gpu if torch.cuda.is_available() else 'cpu')

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        # self.lin1 = torch.nn.Linear(in_channels, out_channels // 2)
        # self.lin2 = torch.nn.Linear(in_channels, out_channels // 2)
        self.lin1 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: x1表示structure属性， x2表示attribute属性
        x = self.lin1(x)
        # x2 = self.lin2(x)
        # x = torch.cat([x1, x2], 1)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j

    # def aggregate(self, x_j, edge_index):
    #     row, col = edge_index
    #     aggr_out = scatter(x_j, row, dim=0, reduce='sum')
    #     return aggr_out

class GCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # bias(如果有的话)
        self.bias = P.Bias
        # 传进来的tensor是否稀疏(暂时不用)
        self.is_sparse = P.TensorSparse
        # 额外向网络添加权重，使so/sr/or按照该权重向加
        self.weight_add = torch.nn.Parameter(torch.randn(3, 1))
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # 定义dropout
        self.dropout = nn.Dropout(P.Drop_Out)
        # 节点集合，用于gcn初始化embedding
        self.nodeSet = [i for i in range(P.Object_Num)]
        self.nodeSet = torch.Tensor(self.nodeSet).to(int)
        self.nodeSet = self.nodeSet.to(device)
        # 关系集合，用于初始化embedding
        self.relationSet = [i for i in range(P.Relation_Num)]
        self.relationSet = torch.Tensor(self.relationSet).to(int)
        self.relationSet = self.relationSet.to(device)
        # 有bias就把它塞进参数里面(暂时不用)
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(P.Emb_Dimension))
        # self.reset_parameters()(暂时不用)
        # 节点和关系的emb函数
        self.node_emb = nn.Embedding(P.Object_Num, P.Emb_Dimension)
        self.relation_emb = nn.Embedding(P.Relation_Num, P.Emb_Dimension)
        self.node_emb.weight.data.normal_(0, 0.01)
        self.relation_emb.weight.data.normal_(0, 0.01)
        # 两层的gcn
        self.gcn1 = GCNConv(P.Emb_Dimension, 128)
        self.gcn2 = GCNConv(128, P.Emb_Dimension)

    def nnfpmc(self, head, edge, tail, node):
        # tf相关代码
        # 模型第一步 将头尾节点以及边（关系）进行emb 并让他们俩俩点乘 若id的list长度为32（batchsize长度），则得到的结果为32*emb长度（向量横着放）
        multi_so = torch.mul(self.tanh(node[head]), self.tanh(node[tail]))
        multi_sr = torch.mul(self.tanh(node[head]), self.tanh(self.relation_emb(edge)))
        multi_or = torch.mul(self.tanh(node[tail]), self.tanh(self.relation_emb(edge)))

        # 将得到的三个值按权重相加
        mid_value = self.weight_add[0] * multi_so + self.weight_add[1] * multi_sr + self.weight_add[2] * multi_or
        # mid_value = multi_so + multi_sr + multi_or

        y_pre = self.sigmoid(torch.sum(mid_value, dim=1))
        y_pre = self.dropout(y_pre)

        return y_pre

    def saveModel(self, edge_index):
        relation = self.relation_emb(self.relationSet)
        node = self.node_emb(self.nodeSet)

        # gcn相关代码
        nodeWithN = self.gcn1(node, edge_index)
        nodeWithN = F.relu(nodeWithN)
        nodeWithN = F.dropout(nodeWithN, p=P.Drop_Out, training=self.training)
        nodeWithN = self.gcn2(nodeWithN, edge_index)
        # dim=1表示对列做softmax
        finalNode = F.softmax(nodeWithN, dim=1)
        torch.save({'node': finalNode, 'edge': relation}, P.GcnEmbedding_FilePath)

    def forward(self, edge_index, head, edge, tail, ng_tail, ng_rela):
        node = self.node_emb(self.nodeSet)

        # gcn相关代码
        nodeWithN = self.gcn1(node, edge_index)
        nodeWithN = F.relu(nodeWithN)
        nodeWithN = F.dropout(nodeWithN, p=P.Drop_Out, training=self.training)
        nodeWithN = self.gcn2(nodeWithN, edge_index)
        # dim=1表示对列做softmax
        finalNode = F.softmax(nodeWithN, dim=1)
        # 头节点 尾节点 关系的embedding表示
        headNode = finalNode[head]
        tailNode = finalNode[tail]
        ng_tailNode = finalNode[ng_tail]
        relation = self.relation_emb(edge)
        ng_relation = self.relation_emb(ng_rela)
        # transe来计算分数
        pos_score = torch.norm(headNode+relation-tailNode, p=P.Norm, dim=1)
        ng_score = torch.norm(headNode+ng_relation-ng_tailNode, p=P.Norm, dim=1)
        # 得到张量分解模型的预测
        pos_ypre = self.nnfpmc(head, edge, tail, finalNode)
        ng_ypre = self.nnfpmc(head, edge, ng_tail, finalNode)
        # return  pos_ypre, ng_ypre

        # return pos_ypre, ng_ypre
        return pos_score, ng_score, pos_ypre, ng_ypre


    def testForward(self, edge_index, head, edge, tail):
        node = self.node_emb(self.nodeSet)
        nodeWithN = self.gcn1(node, edge_index)
        nodeWithN = F.relu(nodeWithN)
        nodeWithN = F.dropout(nodeWithN, p=P.Drop_Out, training=self.training)
        nodeWithN = self.gcn2(nodeWithN, edge_index)
        # dim=1表示对列做log_softmax
        finalNode = F.log_softmax(nodeWithN, dim=1)
        # 头节点 尾节点 关系的embedding表示
        headNode = finalNode[head]
        tailNode = finalNode[tail]
        relation = self.relation_emb(edge)
        score = torch.norm(headNode + relation - tailNode, p=P.Norm, dim=1)

        return score





