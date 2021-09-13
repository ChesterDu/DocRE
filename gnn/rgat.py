import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import activation
from ..model import make_activation
from torch.nn.parameter import Parameter
import dgl.function as fn

class LayerRGAT(nn.Module):
    def __init__(self,node_dim,edge_dim,M,K,activation='relu'):
        ## 
        # M: number of relation attention head
        # K: number of node attention head
        super(LayerRGAT,self).__init__()

        # Hyperparameter
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.M = M
        self.K = K

        # Parameters
        self.node_att_W = Parameter(torch.randn((1,K,1,node_dim,node_dim)))
        self.rel_att_W = Parameter(torch.randn((1,M,1,node_dim,node_dim)))
        self.rel_att_W1 = Parameter(torch.randn((M,edge_dim,node_dim)))
        self.rel_att_b1 = Parameter(torch.randn((M,1,node_dim)))
        self.rel_att_W2 = Parameter(torch.randn((M,node_dim,1)))
        self.rel_att_b2 = Parameter(torch.randn((M,1,1)))

        self.activation = make_activation(activation)

        self.node_out_fc = nn.Sequential(nn.Linear(2 * node_dim,node_dim), self.activation)

        self.node_fc = nn.Sequential(nn.Linear(node_dim,node_dim), self.activation)
        self.edge_fc = nn.Sequential(nn.Linear(edge_dim,edge_dim), self.activation)

        # self.output_gate_W = nn.Linear(node_dim,1)

        # self.relEmb = Parameter(torch.randn((REL_TYPE_NUM,REL_TYPE_EMB_DIM)))
    
    def attention_mech(self,edges):
        # a = F.cosine_similarity(edges.src.data['h'],edges.dst.data['h'])
        a = torch.diag(torch.matmul(edges.src['h'],edges.dst['h'].t()))
        g = torch.sigmoid(torch.matmul(F.relu(torch.matmul(edges.data['h'], self.rel_att_W1) + self.rel_att_b1),self.rel_att_W2) + self.rel_att_b2).transpose(0,1).squeeze(2)  # [edge_num,M]

        return {'a': a, "g":g}

    def message_func(self,edges):
        return {'h_j': edges.dst['h'], 'a':edges.data['a'], 'g':edges.data['g']}
    
    def reduce_func(self,nodes):
        h_j = nodes.mailbox['h_j'].unsqueeze(2).unsqueeze(1) #[Node_num,1,N_i,1,node_dim] [1,K,1,node_dim,node_dim]
        # print(h_j.shape)
        ## compute the node attention
        alpha = F.softmax(nodes.mailbox['a'],dim=1) # [N]
        # print(torch.matmul(h_j,self.node_att_W).shape)
        # print(alpha.shape)
        h_att = torch.sum(torch.sum(alpha.unsqueeze(1).unsqueeze(-1) * torch.matmul(h_j,self.node_att_W).squeeze(3),2),1) / self.K
        # print(h_att.shape)
        ## compute the relation attention
        # print(nodes.mailbox['g'].shape)
        beta = F.softmax(nodes.mailbox['g'].transpose(1,2),dim=2) # [Node_num,M,N_i]
        # print(torch.matmul(h_j,self.rel_att_W).shape)
        h_rel = torch.sum(torch.sum(beta.unsqueeze(-1) * torch.matmul(h_j,self.rel_att_W).squeeze(3),2),1) / self.M

        h = self.node_out_fc(torch.cat((h_att,h_rel),dim=1))

        return {'h':h}
    def forward(self,g,node_features,edge_features):
        g.ndata['h'] = node_features
        g.edata['h'] = edge_features

        g.apply_edges(self.attention_mech)
        g.update_all(self.message_func,self.reduce_func)

        out_features = g.ndata.pop('h')

        new_features = self.node_fc(out_features) + node_features # residual connection
        edge_features = self.edge_fc(edge_features) + edge_features

        # out_gate = torch.sigmoid(self.output_gate_W(node_features))
        # new_features = out_gate * out_features + (1-out_gate) * node_features

        return new_features, edge_features


class multiLayerRGAT(nn.Module):
    def __init__(self,node_in_dim,node_dim,node_out_dim,edge_in_dim,edge_dim,M,K,L,activation='relu'):
        super(multiLayerRGAT,self).__init__()
        self.activation = make_activation(activation)
        self.RGATS = nn.ModuleList([LayerRGAT(node_dim,edge_dim,M,K,activation) for _ in range(L)])
        self.node_in_fc = nn.Sequential(nn.Linear(node_in_dim,node_dim),self.activation)
        self.edge_in_fc = nn.Sequential(nn.Linear(edge_in_dim,edge_dim),self.activation)
        self.node_out_fc = nn.Sequential(nn.Linear(node_dim,node_out_dim),self.activation)
        self.L = L
    def forward(self,g,node_features,edge_features):
        node_features = self.node_in_fc(node_features)
        edge_features = self.edge_in_fc(edge_features)
        for RGAT in self.RGATS:
          node_features,edge_features =  RGAT(g,node_features,edge_features)
        
        node_features = self.node_out_fc(node_features)
        return node_features
