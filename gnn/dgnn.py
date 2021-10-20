import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from gnn.rgat import LayerRGAT

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def structureAwareAttention(query, key, value, structure_k, structure_v,dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query, key, value [h, nnode, dk]
    # sturcture_k, structure_v [nnode, nnode, dk]
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k) # [h, nnode, nnode]
    scores_k = torch.matmul(query.transpose(0,1),structure_k.transpose(1,2)) / math.sqrt(d_k)
    scores = scores + scores_k.transpose(0,1)
    p_attn = F.softmax(scores, dim = -1) #[h, nnode, nnode]
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value) + torch.matmul(p_attn.transpose(0,1),structure_v).transpose(0,1), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_edge, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.linears_structure = clones(nn.Linear(d_edge, self.d_k), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x, structure):

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        nnode = x.shape[0]
        query, key, value = \
            [l(x).view(nnode, self.h, self.d_k).transpose(0, 1)
             for l, x in zip(self.linears, (x, x, x))]
        
        query,key = self.activation(query),self.activation(key)
        structure_k, structure_v = self.activation(self.linears_structure[0](structure)), self.linears_structure[1](structure)

        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = structureAwareAttention(query, key, value, structure_k, structure_v, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(0, 1).contiguous() \
             .view(nnode, self.h * self.d_k)
        return self.linears[-1](x)

class structureAwareGraphEncoder(nn.Module):
    def __init__(self,layers,h,d_model,d_edge,edge_num,dropout):
        super(structureAwareAttention,self).__init__()
        self.L = layers
        self.layers = clones(MultiHeadedAttention(h,d_model,d_edge,dropout),layers)
        self.edge_embeds = nn.Embedding(edge_num,d_edge)

    def forward(self,x,adjMatrix):
        structure = self.edge_embeds(adjMatrix)
        x_lst = [x.unsqueeze(0)]
        for layer in self.layers:
            x = layer(x,structure)
            x_lst.append(x.unsqueeze(0))

        return x_lst


class GRUGraphFuser(nn.Module):
    def __init__(self,h_model,dropout=0.1):
        super(GRUGraphFuser,self).__init__()
        self.gru = nn.GRU(h_model,h_model,1,dropout=dropout)
        self.linear_z = nn.Linear(4*h_model,h_model)
        self.activation = nn.Sigmoid()

    def forward(self,a_lst,b_lst):
        # a_lst, b_lst: list(Tensor[nnode,h_model])
        a_lst = torch.cat(a_lst,dim=0) #[L, nnode, h_model]
        b_lst = torch.cat(b_lst,dim=0) #[L, nnode, h_model]
        z = self.activation(self.linear_z(torch.cat([a_lst,b_lst,a_lst * b_lst, a_lst - b_lst],dim=-1))) #[L, nnode, h_model]
        fuse_ab = z * a_lst + (1-z) * b_lst #[L, nnode, h_model]

        _,hn = self.gru(fuse_ab)

        return hn.squeeze(0)
