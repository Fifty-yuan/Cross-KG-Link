import torch
import loadData as LD
import Parameter as P

from gcn import GCN

import test_utils as tu

def evaluate():
    print('test start...')

    test_total = len(test_twoKB)

    twoKB_edge_mr = 0
    twoKB_hit1 = 0
    twoKB_hit3 = 0
    twoKB_hit5 = 0
    twoKB_hit10 = 0
    oneKB_edge_mr = 0
    oneKB_hit1 = 0
    oneKB_hit3 = 0
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
        if i % 500 == 0:
            print(golden_triple[:3], end=': ')
            print('edge_pos=' + str(edge_pos), end=', ')
            print('test ---' + str(i) + '--- triple')
            print(i, end="\r")
            # print('edge_pos_filter=' + str(edge_pos_filter), end=', ')
        # 所有排名相加，过滤排名相加 最后再算平均
        twoKB_edge_mr += edge_pos
        if edge_pos <= 1:
            twoKB_hit1 += 1
        if edge_pos <= 3:
            twoKB_hit3 += 1
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
        if edge_pos <= 3:
            oneKB_hit3 += 1
        if edge_pos <= 5:
            oneKB_hit5 += 1
        if edge_pos <= 10:
            oneKB_hit10 += 1
    # 算平均
    twoKB_edge_mr /= test_total
    twoKB_hit1 /= test_total
    twoKB_hit3 /= test_total
    twoKB_hit5 /= test_total
    twoKB_hit10 /= test_total
    oneKB_edge_mr /= test_total
    oneKB_hit1 /= test_total
    oneKB_hit3 /= test_total
    oneKB_hit5 /= test_total
    oneKB_hit10 /= test_total

    # total
    total_edge_mr = (twoKB_edge_mr + oneKB_edge_mr) / 2
    total_hit1 = (twoKB_hit1 + oneKB_hit1) / 2
    total_hit3 = (twoKB_hit3 + oneKB_hit3) / 2
    total_hit5 = (twoKB_hit5 + oneKB_hit5) / 2
    total_hit10 = (twoKB_hit10 + oneKB_hit10) / 2

    # \t代表空格
    print('\t\t\tmean_rank\t\t\t')
    # %代表格式化输出 .3f控制小数点后三位
    print('total_edge(raw)\t\t\t%.3f\t\t\t' % total_edge_mr)
    print('total_hit@1\t\t\t%.3f\t\t\t' % total_hit1)
    print('total_hit@1\t\t\t%.3f\t\t\t' % total_hit3)
    print('total_hit@5\t\t\t%.3f\t\t\t' % total_hit5)
    print('total_hit@10\t\t\t%.3f\t\t\t' % total_hit10)
    print('twoKB_edge_mr(raw)\t\t\t%.3f\t\t\t' % twoKB_edge_mr)
    print('twoKB_hit@1\t\t\t%.3f\t\t\t' % twoKB_hit1)
    print('twoKB_hit@1\t\t\t%.3f\t\t\t' % twoKB_hit3)
    print('twoKB_hit@5\t\t\t%.3f\t\t\t' % twoKB_hit5)
    print('twoKB_hit@10\t\t\t%.3f\t\t\t' % twoKB_hit10)
    print('oneKB_edge_mr(raw)\t\t\t%.3f\t\t\t' % oneKB_edge_mr)
    print('oneKB_hit@1\t\t\t%.3f\t\t\t' % oneKB_hit1)
    print('oneKB_hit@1\t\t\t%.3f\t\t\t' % oneKB_hit3)
    print('oneKB_hit@5\t\t\t%.3f\t\t\t' % oneKB_hit5)
    print('oneKB_hit@10\t\t\t%.3f\t\t\t' % oneKB_hit10)

if __name__ == '__main__':
    # 设置随机种子
    LD.set_random_seed(P.seed)
    # 获取 edge_index/训练集三元组/测试集三元组/测试时所用的训练三元组
    edge_index, train_data, test_data, trainTup = LD.Load_data()

    test_twoKB = test_data[:20000]
    test_oneKB = test_data[20000:]

    # Gpu训练
    device = torch.device(P.Gpu if torch.cuda.is_available() else 'cpu')
    # 载入模型
    gcnModel = GCN().to(device)
    # if len(P.Gpu_id) > 1:
    #     model = nn.DataParallel(model, device_ids=P.Gpu_id)
    gcnModel.load_state_dict(torch.load(P.GcnModel_FilePath))

    evaluate()