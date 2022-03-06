import time

import torch
import torch as th
import torch.nn.functional as F

import dgl
import numpy as np

from SGdataset import load_arxiv_data, loadSubGraph
from SGmodels import SAGE
from SGutils import spilt_data, loss_softlabel, evaluate

if th.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'



teacher_path = "./model/model.pt"

best_SG_path = {
    '005' : './model/SG_model_005.pt',
    '01' : './model/SG_model_01.pt',
    '05' : './model/SG_model_05.pt',
    '07' : './model/SG_model_07.pt',
    'teacher_mini' : './model/model.pt',
    'teacher_full' : './model/fullBatchModel.pt'
    }

distill_SG_path = {
    '005' : './model/SG_distill_005.pt',
    '01' : './model/SG_distill_01.pt',
    '05' : './model/SG_distill_05.pt',
    '07' : './model/SG_distill_07.pt',
    }


#输入的要素，图，特征，标签，训练集，测试集，输入特征的维数，输出类别的维数
def student_train(g, features, labels,
                  train_mask, validation_mask, test_mask,
                  in_feat ,out_class, distill_layer, loss_weight, path_t):

    temp_validation = 0
    best_score = 0
    best_loss = 1000.0


    #首先先处理需要跑的子图数据
    g.add_edges(g.nodes(), g.nodes())
    g = g.to(device=device)
    features = features.to(device=device)
    labels = labels.to(device=device)



    #定义教师和学生网络
    s_model = SAGE(in_feats=in_feat, hid_feats=128, out_class=out_class).to(device=device)
    t_model = SAGE(in_feats= in_feat, hid_feats=128, out_class=out_class)
    t_model.load_state_dict(th.load(path_t))
    t_model.to(device)


    optimizer = th.optim.Adam(s_model.parameters(), lr=0.01)


    for epoch in range(250):
        s_model.train()
        loss_list = []
        additional_loss_list = []
        t0 = time.time()


        """
        t_model--→model_dict[][]--→auxilary_model.collect_model--→utils.get_teacher
        --→定义了一个GAT网络--→gat.GAT，forward的return，当middle=False，返回输出的logits，
        当middle=True，返回logits和第一层的输出middle_feat
        
        转auxilary_model的local函数详解：
        关于local_model和local_model_s,转get_local_model
        local_structure.get_local_model: return distanceNet(),
            转local_structure.distanceNet:
        继承nn.Module,该模型没有层数，直接调用forward进行运算。
        forward(graph, feats):
        对graph，使用graph.local_var()把图中的运算固定在本函数内，不会影响原本的图。
        对feats，使用feats.view函数，对feats的张量进行重组, 先将张量拍平成一维，
        然后再通过参数重构维度。此处的view(-1, 1, feats.shape[1])
        先把整个feat拍平，然后化为1*1*shape[1]的张量，即
        [      [[shape[1]数量的值 1,2,3 等等  ], 
                [                           ],
                [                           ]]
        ]   这样一个张量。然后将其保存至graph.ndata中。
        fn.u_sub_v用于计算一条边上的信息，通过把u和v之间的feat相减并作为输出，
        即ftl和ftr的差diff，在图上生成边，apply_edges.
        使用pop删除diff并保存在e中，然后做一个运算，应该就是论文中的那个公式。
        然后放入edge_softmax中计算并返回
        
        
        此处，隔壁的main第60行，取得minibatch后的子图及相关数据feat， label
            81行：通过generate_label函数生成教师网络的softlabel标签logits_t
            转utils.generate_label:
        教师网络运行mini batch生成的子图，返回输出的logits_t
        此时将得到的logits_t and logits放入loss_fn_kd中，得到损失函数class_loss
            转auxilary_loss.loss_fn_kd:
        ce_loss设置为BCEWithLogitsLoss，
        利用torch.where 来筛选labels_t，此处的意思是,通过logits_t.shape,构造对应大小的张量，
        如果对应位置的logits_t>0则对应位置置1，否则就对应位置置0。
        将学生网络学习到的logits和0,1化的教师网络logits做损失函数并返回
            取得class_loss后，对class_loss做detach操作，保证每一次的loss都不下降
        上面的这个loss_fn_kd在mi模式下完全没有用到，麻了。
        
            target-layer = 2,即此处使用学生网络的第二层进行学习。
        mi_loss = gen_mi_loss()
        params: 见auxilary_loss.gen_mi_loss函数介绍部分
        通过输入的middle_feats_s和现场生成的middle_feats_t, 分别得出一个图中两点之间的差值
        dist_t, dist_s, 然后输入graph_KLDiv中计算每个边的KL损失。
        
        重点：使用graph.update_all(fn.u_mul_e('kldiv', 'diff', 'm'), fn.sum('m', 'kldiv'))
        直接更新节点内的ndata，即消息函数fn.u_mul_e, 更新函数sum，
        通过将源节点特征 kldiv 与边特征 diff 相乘生成消息 m， 然后对所有消息求和来更新节点特征 kldiv
        然后对kldiv进行flatten和mean操作并返回
        KL计算完毕以后即为mi_loss, 乘上权重加到原loss上， 然后进行优化
            
        """

        logits, h_feat = s_model(g, features)

        #原本单训练一个学生网络所使用的loss
        # loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        ce_loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        soft_loss = loss_softlabel(t_model, h_feat[distill_layer], g, features, distill_layer, device)

        additional_loss = loss_weight*soft_loss

        loss = (1-loss_weight)*ce_loss + additional_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        additional_loss_list.append(additional_loss.item() if additional_loss!=0 else 0)

        loss_data = np.array(loss_list).mean()
        additional_loss_data = np.array(additional_loss_list).mean()

        print(f"Epoch {epoch:05d} | Loss: {loss_data:.4f} | Mi: {additional_loss_data:.4f} | Time: {time.time() - t0:.4f}s")


        validation_score = evaluate(s_model, g, features, labels, validation_mask)
        if validation_score > temp_validation or loss_data < best_loss:
            temp_validation = validation_score
            best_loss = loss_data
            test_score = evaluate(s_model, g, features, labels, test_mask)
            if test_score > best_score:
                best_score = test_score
                print(f"Best_epoch: {epoch:04d} | score: {best_score:.4f}")
                th.save(s_model.state_dict(), save_path)






        # print("Epoch {:05d}  |  Loss {:.4f}  |   Test Acc {:.4f}".format(epoch, loss.item(), acc))


def test_teacher(graph, features, labels, test_mask, in_feat, out_class, path_t):

    graph.add_edges(graph.nodes(), graph.nodes())
    graph = graph.to(device=device)
    features = features.to(device=device)
    labels = labels.to(device=device)
    # 尝试直接加载教师网络跑一次数据，看看结果如何
    #此处的教师网络没有使用state_dict的方法进行保存，出现了一些问题，建议回头重新保存一次
    t_model = SAGE(in_feats= in_feat, hid_feats=128, out_class=out_class)
    t_model.load_state_dict(th.load(path_t))
    t_model.to(device)

    #evaluate所需输入: model, graph, features, labels, mask
    acc = evaluate(t_model, graph, features, labels, test_mask)
    print("Test Acc  {:.4f}".format(acc))



'''
当你训练时，记得修改以下几个点
1. loss weight  ----0，0.5，0.99
2. subgraph     ----载入图时
3. graph model  ----测试在本模型在其他集上面的效果→ 234、237
4. distill layer----
'''

distill_layer = 1
loss_weight = 0.999
path_sub = "./data/ogbn_arxiv/subgraph/subgraph_005.dgl"

if loss_weight == 0:
    save_path = best_SG_path['005']
    test_path = best_SG_path['teacher_full']
else:
    save_path = distill_SG_path['005']
    test_path = distill_SG_path['005']




#图, 输入特征的维数, 输出类别的维数, 训练集, 验证集, 测试集，蒸馏教师网络的第几层，蒸馏函数的损失权重
graph, in_feat, out_class, train_nids, valid_nids, test_nids = load_arxiv_data()


#对原图中的数据进行数据处理
#训练函数输入时的要素，图✓，特征，标签，训练集✓，验证集✓，测试集✓，输入特征的维数✓，输出类别的维数✓，
#蒸馏教师网络的第几层✓，蒸馏函数的损失权重✓，教师网络的路径✓
g_feature = graph.ndata['feat']
g_label = graph.ndata['label']



subgraph, list_subg_id, list_train_nids, list_validation_nids, list_test_nids = loadSubGraph(path_sub, train_nids, valid_nids, test_nids)


# 其中NID指子图中的节点原本在图中的节点号，id是指子图中的节点在子图中的节点号
train_NID, train_id = spilt_data(list_subg_id, list_train_nids)

validation_NID, validation_id = spilt_data(list_subg_id, list_validation_nids)

test_NID, test_id = spilt_data(list_subg_id, list_test_nids)


sg_label = subgraph.ndata['label']
sg_nfeat = subgraph.ndata['feat']

print(subgraph)

th.manual_seed(42)
th.cuda.manual_seed(42)


#输入的要素, 图, 特征, 标签, 训练集, 验证集, 测试集, 输入特征的维数, 输出类别的维数, 蒸馏教师网络的第几层, 蒸馏函数的损失权重，教师网络的路径
# student_train(subgraph, sg_nfeat, sg_label, train_id, validation_id, test_id, in_feat, out_class, distill_layer, loss_weight, teacher_path)

#载入模型在原图上的测试集跑出结果
test_teacher(graph, g_feature, g_label, test_nids, in_feat, out_class, test_path)

#测试模型在子集上的效果
# test_teacher(subgraph, sg_nfeat, sg_label, test_id, in_feat, out_class, test_path)


#输入的要素, 图, 特征, 标签, 训练集, 验证集, 测试集, 输入特征的维数, 输出类别的维数, 蒸馏教师网络的第几层, 蒸馏函数的损失权重，教师网络的路径
#用于训练一个fullbatch的教师网络，来看看效果
# student_train(graph, g_feature, g_label, train_nids, valid_nids, test_nids, in_feat, out_class, distill_layer, loss_weight, teacher_path)








