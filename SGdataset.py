import dgl
import torch

from ogb.nodeproppred import DglNodePropPredDataset

#尽量别在其他的本目录下引用其他.py,防止发生死锁


def load_arxiv_data():
    dataset = DglNodePropPredDataset('ogbn-arxiv')
    graph, node_labels = dataset[0]
    # Add reverse edges since ogbn-arxiv is unidirectional. 对这一步持有怀疑态度
    graph.ndata['label'] = node_labels[:, 0]
    # print(graph)
    # print(node_labels)

    node_features = graph.ndata['feat']
    num_features = node_features.shape[1]
    num_classes = (node_labels.max() + 1).item()
    # print('Number of classes:', num_classes)

    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']

    print("data loaded!")

    # return graph, node_labels, node_features, num_features, num_classes, train_nids, valid_nids, test_nids
    return graph, num_features, num_classes, train_nids, valid_nids, test_nids


#载入子图，并且生成子图的相关参数
def loadSubGraph(path, train_nids, validation_nids, test_nids):
    gload = dgl.load_graphs(path)
    glist = gload[0]

    subg = glist[0]

    subg_id = subg.ndata[dgl.NID]

    # 测试对比两个数组，找到相同元素
    list_subg_id = subg_id.numpy()
    list_train_nids = train_nids.numpy()
    list_validation_nids = validation_nids.numpy()
    list_test_nids = test_nids.numpy()

    return subg, list_subg_id, list_train_nids, list_validation_nids, list_test_nids


