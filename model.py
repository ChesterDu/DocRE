import torch.nn.functional as F
import copy
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


ENTITY_TYPE_NUM = 7
ENTITY_TYPE_EMB_DIM = 768
REL_TYPE_NUM = 15
REL_TYPE_EMB_DIM = 768
OUTPUT_NUM = 97

class LayerRGAT(nn.Module):
    def __init__(self,node_dim,edge_dim,M,K):
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

        self.node_out_fc = nn.Linear(2 * node_dim,node_dim)

        self.node_fc = nn.Linear(node_dim,node_dim)
        self.edge_fc = nn.Linear(edge_dim,edge_dim)

        self.output_gate_W = nn.Linear(node_dim,1)

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
        h = F.relu(h)

        return {'h':h}
    def forward(self,g,node_features,edge_features):
        g.ndata['h'] = self.node_fc(node_features)
        g.edata['h'] = self.edge_fc(edge_features)

        g.apply_edges(self.attention_mech)
        g.update_all(self.message_func,self.reduce_func)

        out_features = g.ndata.pop('h')
        out_gate = torch.sigmoid(self.output_gate_W(node_features))
        new_features = out_gate * out_features + (1-out_gate) * node_features

        return new_features


class multiLayerRGAT(nn.Module):
    def __init__(self,node_in_dim,node_dim,node_out_dim,edge_in_dim,edge_dim,M,K,L):
        super(multiLayerRGAT,self).__init__()
        self.RGATS = nn.ModuleList([LayerRGAT(node_dim,edge_dim,M,K) for _ in range(L)])
        self.node_in_fc = nn.Sequential(nn.Linear(node_in_dim,node_dim),nn.LeakyReLU())
        self.edge_in_fc = nn.Sequential(nn.Linear(edge_in_dim,edge_dim),nn.LeakyReLU())
        self.node_out_fc = nn.Sequential(nn.Linear(node_dim,node_out_dim),nn.LeakyReLU())
        self.L = L
    def forward(self,g,node_features,edge_features):
        node_features = self.node_in_fc(self.node_in_fc)
        edge_features = self.edge_in_fc(edge_features)
        for RGAT in self.RGATS:
          node_features =  RGAT(g,node_features,edge_features)
        
        node_features = self.node_out_fc(node_features)
        return node_features


class finalModel(nn.Module):
    def __init__(self,node_in_dim,node_dim,node_out_dim,edge_in_dim,edge_dim,M,K,L,model_name='bert-base-uncased',max_token_len = 256,max_sen_num = 10):
        super(finalModel,self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_token_len = max_token_len
        self.max_sen_num = max_sen_num
        # self.model = BertModel(self.config)
        self.encoder_config = BertConfig.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)

        # GNNs
        self.gnn = multiLayerRGAT(node_in_dim,node_dim,node_out_dim,edge_in_dim,edge_dim,M,K,L)
        self.entityTypeEmb = Parameter(torch.randn((ENTITY_TYPE_NUM,ENTITY_TYPE_EMB_DIM)))
        self.relEmb = Parameter(torch.randn((REL_TYPE_NUM,REL_TYPE_EMB_DIM)))

        # Prediction Layer
        self.pred_fc = nn.Linear(2*node_dim,OUTPUT_NUM)

    def token2id(self,sents):
    # sents is a list of list of tokens
        token_ids = []
        sen_start_pos_lst = [0]   # starting ids of each sentence 
        for i,sen in enumerate(sents):
            temp = self.tokenizer.encode(sen[:-1])  # remove the <ROOT> Token
            if i != 0:
                temp = temp[1:] # remove the [CLS]

            token_ids += temp
            start_pos = sen_start_pos_lst[-1] + len(temp)
            sen_start_pos_lst.append(start_pos)

        token_ids += [0] * (self.max_token_len - len(token_ids))
        sen_start_pos_lst = sen_start_pos_lst[:-1]
        sen_start_pos_lst += [-1] * (self.max_sen_num - len(sen_start_pos_lst))
        return token_ids, sen_start_pos_lst
    
    def contextual_encoding(self,batched_token_ids):
        h = self.encoder(input_ids = batched_token_ids)[0]  #[batch, seq_len, hidden_len]
        return h
    
    def forward(self,g,node_features,edge_features,head_ent_nodes,tail_ent_nodes):
        node_h = self.gnn(g,node_features,edge_features)
        head_ent_h = node_h[head_ent_nodes]
        tail_ent_h = node_h[tail_ent_nodes]

        out = self.pred_fc(torch.cat([head_ent_h,tail_ent_h],dim=1))

        return out
