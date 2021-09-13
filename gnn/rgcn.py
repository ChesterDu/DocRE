## Adapted from DGL RGCN Implementation
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import dgl.function as fn


class LayerRGCN(nn.Module):
    def __init__(self,in_feat,out_feat,rel_num,activation):
        super(LayerRGCN,self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_num = rel_num
        self.activation = activation

        self.weight = Parameter(torch.Tensor(rel_num,in_feat,out_feat))
        self.W_0 = Parameter(torch.Tensor(in_feat,out_feat))
    
    def msg_func(self,edges):
        W = self.weight[edges.data['edge_id']]
        print(W.shape,edges.src['h'].shape)
        msg = torch.bmm(edges.src['h'].unsqueeze(1),W).squeeze(1)
        print(msg.shape,edges.data['norm'].shape)
        msg = msg * edges.data['norm'].unsqueeze(-1)

        return {'msg':msg}

    def apply_func(self,nodes):
        h_sum = nodes.data['h_sum']
        h = nodes.data['h']
        out = torch.mm(h,self.W_0) + h_sum
        out = self.activation(out)
        return {'h':out}

    def forward(self,g):
        g.update_all(self.msg_func,fn.sum(msg='msg',out='h_sum'),self.apply_func)


class multiLayerRGCN(nn.Module):
    def __init__(self,node_in_dim,node_dim,node_out_dim,rel_num,L,activation):
        super(multiLayerRGCN,self).__init__()
        self.L = L
        self.node_in_dim = node_in_dim
        self.node_dim = node_dim
        self.node_out_dim = node_out_dim
        self.node_in_fc = nn.Linear(node_in_dim,node_dim)
        self.node_out_fc = nn.Linear(node_dim,node_out_dim)
        self.gnns = nn.ModuleList([LayerRGCN(node_dim,node_dim,rel_num,activation) for _ in range(L)])

    def forward(self,g,node_in_feat):
        h = F.relu(self.node_in_fc(node_in_feat))
        g.ndata['h'] = h

        for layer in self.gnns:
            layer(g)
        
        out_feat = g.ndata.pop('h')
        out = F.relu(self.node_out_fc(out_feat))
        
        return out

