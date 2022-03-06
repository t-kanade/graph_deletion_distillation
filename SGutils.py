import numpy
import numpy as np
import numpy.random as nrd
import dgl
import dgl.function as fn
from dgl.data.utils import save_graphs, load_graphs
from dgl.nn.pytorch.softmax import edge_softmax

import torch as th
import torch.nn as nn

from sklearn.metrics import f1_score



# from SGdataset import load_arxiv_data, loadSubGraph

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits, h_feat = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        print(indices.data)
        print(labels.data.cpu().numpy())
        correct = th.sum(indices == labels)
    model.train()
    return correct.item() * 1.0 / len(labels)



#设置一个删点的工具
def randomDeletion(graph, delete_num):
    #remove_nodes这个函数，删除这个点，
    nrd.seed(1)
    delete_nodes = nrd.randint(low=0, high=169343, size=delete_num)
    delete_nodes = numpy.sort(delete_nodes)     #把删点的序列排序，从小到大删除
    #delete the nodes
    for i in range(0, delete_num):
        temp_nodes = delete_nodes[i] - i
        graph = dgl.remove_nodes(graph, temp_nodes)

    print(graph)
    print('Successfully delete {} nodes!'.format(delete_num))
    return graph

#子图生成工具
def createSubGraph(graph, reserve_num, name):
    nrd.seed(1)
    subg_nodes = nrd.choice(169343, reserve_num, replace=False)
    subg_nodes = numpy.sort(subg_nodes)     # 把删点的序列排序，从小到大删除
    subg = graph.subgraph(subg_nodes)       #生成子图
    print(subg)

    save_graphs("./data/ogbn_arxiv/subgraph/"+name, subg)
    print("you can see the subgraph in the folder 'dataset'   ~~")
    return

#对比子图的节点的NID和对应训练集的NID，找出相同项，生成新的训练集
def spilt_data(subg_id, set_id):
    subg_data = np.intersect1d(subg_id, set_id, return_indices=True)

    subg_NID = th.from_numpy(subg_data[0])       #NID，即对应训练集在原本大图中的节点编号
    subg_id = th.from_numpy(subg_data[1])        #id，即对应训练集在子图中的节点编号
    return subg_NID, subg_id


def loss_softlabel(model_t, feats_s, subgraph, feats, distill_layer, device):
    """
    the same function as the gen_mi_loss in the Lsp
    :param model_t: 教师网络模型，一个graphSAGE，输入graph和feat进行forward，输出最后的结果h和第一层的softlabel
    :param feats_s:学生网络学习到的中间层softlabel
    :param subgraph: 送给教师网络运行的子图
    :param feats:   送给教师网络运行的特征
    :return:    进过KL散度比较后的损失函数
    """
    with th.no_grad():
        t_model_g = subgraph
        _, feats_t = model_t(subgraph, feats)
        feats_t = feats_t[2]


    local_model = get_local_model(1);    local_model.to(device)

    dist_t = local_model(subgraph, feats_t)
    dist_s = local_model(subgraph, feats_s)

    graphKL_loss = graph_KLDiv(subgraph, dist_s, dist_t)
    return graphKL_loss



def get_local_model(feat_info):
    '''
    '''

    return distanceNet()

class distanceNet(nn.Module):
    def __init__(self):
        super(distanceNet, self).__init__()

    def forward(self, graph, feats):
        graph = graph.local_var()
        feats = feats.view(-1, 1, feats.shape[1])
        graph.ndata.update({'ftl': feats, 'ftr': feats})
        # compute edge distance

        # gaussion
        graph.apply_edges(fn.u_sub_v('ftl', 'ftr', 'diff'))
        e = graph.edata.pop('diff')
        e = th.exp((-1.0 / 100) * th.sum(th.abs(e), dim=-1))

        # compute softmax
        e = edge_softmax(graph, e)
        return e

def graph_KLDiv(graph, edge_s, edge_t):
    '''
    compute the KL loss for each edges set, used after edge_softmax
    '''
    with graph.local_scope():
        nnode = graph.number_of_nodes()
        graph.ndata.update({'kldiv': th.ones(nnode,1).to(edge_s.device)})
        diff = edge_t*(th.log(edge_t)-th.log(edge_s))
        graph.edata.update({'diff':diff})
        graph.update_all(fn.u_mul_e('kldiv', 'diff', 'm'),
                            fn.sum('m', 'kldiv'))

        return th.mean(th.flatten(graph.ndata['kldiv']))





# #创建子图
# reserve_num_005 = 8478
# reserve_num_01 = 16935
# reserve_num_05 = 84672
# reserve_num_07 = 118540
# num_nodes = 169343
#
# graph, num_features, num_classes, train_nids, valid_nids, test_nids = load_arxiv_data()
#
# subgraph_05 = createSubGraph(graph, reserve_num_07, "subgraph_07.dgl")



#下面此处为试验方法是否可行的部分
#
# graph, node_labels, node_features, num_features, num_classes, train_nids, valid_nids, test_nids = load_arxiv_data()
# print(graph)
#
# path = './dataset/subgraph/subgraph_07.dgl'
#
#
# subg, subg_torch_train, subg_torch_validation, subg_torch_test = loadSubGraph(path, train_nids, valid_nids, test_nids)




'''整理出的思路 
1. 创建这个子图，因为有dgl.NID的存在，我们可以直接获得子图node的id和原本图中id的对应关系。 
2. 通过这个关系，通过for i in range(),生成子图对应的train, validation, test集合。 
3.
'''

# #读取原数据并进行操作的思路，废案
# names = []
#
# for i in range(0, 128):
#     names.append(i)
#
# nodes_data = pd.read_csv('./data/ogbn_arxiv/node-feat.csv', names=names)
#
# index_num = nodes_data.index
# print(index_num)
#
# row_data0 = nodes_data.iloc[0]
# list_data0 = row_data0.to_list()
# arr_data0 = np.array(list_data0)
# tensor_data0 = th.from_numpy(arr_data0)
#
# row_data1 = nodes_data.iloc[1]
# list_data1 = row_data1.to_list()
# arr_data1 = np.array(list_data1)
# tensor_data1 = th.from_numpy(arr_data1)
#
# nodes_feat = th.stack((tensor_data0,tensor_data1))
# print(nodes_feat)

# arr_data_0 = np.array(list_data_0)
#
# print(arr_data_0)
# print(type(arr_data_0))

# tensor_data_0 = th.from_numpy(arr_data_0)
#
# print(tensor_data_0)
# print(type(tensor_data_0))

