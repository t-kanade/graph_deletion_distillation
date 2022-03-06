# define the model and the whole network
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F

# we use SAGEConv as out model layer
# we define two conv layer in out SAGE
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_class):
        super(SAGE, self).__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_class, aggregator_type='mean')
        self.h_feat = hid_feats

    def forward(self, graph, inputs):
        # the input are features of nodes
        middle_feat = []
        h = self.conv1(graph, inputs)
        middle_feat.append(h)
        h = F.relu(h)
        middle_feat.append(h)
        h = self.conv2(graph,h)
        middle_feat.append(h)
        return h, middle_feat


def loss_fn_kd(logits, logits_t):
    """
    :param logits: 学生网络在过全连接层之前的输出，即学生网络的softlabel
    :param logit_t: 教师网络过全连接层之前的输出，教师网络的softlabel
    :return: 二者之差形成的损失函数
    """
    ce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    labels_t = torch.where(logits_t>0.0,
                           torch.ones(logits_t.shape).to(logits_t.device),
                           torch.zeros(logits_t.shape).to(logits_t.device))
    ce_loss = ce_loss_fn(logits, labels_t)
    return ce_loss


