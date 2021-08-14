import torch.nn.functional as F
import copy
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from data import ner2id
from fastNLP.embeddings import BertEmbedding, ElmoEmbedding


ENTITY_TYPE_NUM = 7
ENTITY_TYPE_EMB_DIM = 768
REL_TYPE_NUM = 15
REL_TYPE_EMB_DIM = 768
OUTPUT_NUM = 97

def make_embed(vocab,embed_type,embed_pth):
    if embed_type=='bert-base':
        embed = BertEmbedding(vocab, model_dir_or_name=embed_pth,auto_truncate=True,layers='-1', pool_method='avg')
    if embed_type=='Elmo':
        embed = ElmoEmbedding(vocab, model_dir_or_name=embed_pth, requires_grad=True, layers='mix')
        
    node_in_dim = embed.embedding_dim
    return embed,node_in_dim



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
        return {'h_j': edges.src['h'], 'a':edges.data['a'], 'g':edges.data['g']}
    
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
        node_features = self.node_in_fc(node_features)
        edge_features = self.edge_in_fc(edge_features)
        for RGAT in self.RGATS:
          node_features =  RGAT(g,node_features,edge_features)
        
        node_features = self.node_out_fc(node_features)
        return node_features


class finalModel(nn.Module):
    def __init__(self,embed,node_in_dim,node_dim,node_out_dim,edge_in_dim,edge_dim,M,K,L):
        super(finalModel,self).__init__()

        # Model HyperParameters
        self.node_in_dim = node_in_dim
        self.node_dim = node_dim
        self.node_out_dim = node_out_dim
        self.edge_in_dim = edge_in_dim
        self.edge_dim = edge_dim
        self.M = M
        self.K = K
        self.L = L

        # Embedding Layer
        self.embed = copy.deepcopy(embed)

        # GNNs
        self.gnn = multiLayerRGAT(node_in_dim,node_dim,node_out_dim,edge_in_dim,edge_dim,M,K,L)
        self.entityTypeEmb = Parameter(torch.randn((node_in_dim,node_in_dim)))
        self.relEmb = Parameter(torch.randn((REL_TYPE_NUM,REL_TYPE_EMB_DIM)))

        # Prediction Layer
        self.pred_fc = nn.Linear(2*node_out_dim,OUTPUT_NUM)

        self.init_params()
    
    def init_params(self):
        for p in self.gnn.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.normal_(p)
        torch.nn.init.normal_(self.entityTypeEmb)
        torch.nn.init.normal_(self.relEmb)
        for p in self.pred_fc.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.normal_(p)
    
    def contextual_encoding(self,batched_token_ids):
        h = self.embed(batched_token_ids)  #[batch, seq_len, hidden_len]
        return h

    def init_node_edge_features(self,x,sample):
        device = x.device
        node_features = torch.zeros((len(sample['graphData']['nodes']),self.node_in_dim)).to(device)
        sen_start_pos_lst = sample['senStartPos']
        for i,node_data in sample['graphData']['nodes']:
            if node_data['attr'] == 'amr':
                sen_start_pos = sen_start_pos_lst[node_data['sent_id']]
                span_rep_lst = torch.zeros((node_data['pos'][1] - node_data['pos'][0],self.node_in_dim)).to(device)
                for i,pos_id in enumerate(range(node_data['pos'][0],node_data['pos'][1])):
                    span_rep_lst[i] = x[sen_start_pos + pos_id]
                node_feature = torch.sum(span_rep_lst,dim=0) / len(span_rep_lst)
            elif node_data['attr'] == 'mention':
                sen_start_pos = sen_start_pos_lst[node_data['sent_id']]
                span_rep_lst = torch.zeros((node_data['pos'][1] - node_data['pos'][0],self.node_in_dim)).to(device)
                for i,pos_id in enumerate(range(node_data['pos'][0],node_data['pos'][1])):
                    span_rep_lst[i] = x[sen_start_pos + pos_id]
                node_feature = torch.sum(span_rep_lst,dim=0) / len(span_rep_lst)
                node_feature += self.entityTypeEmb[ner2id[node_data['ent_type']]]
            elif node_data['attr'] == 'entity':
                node_feature = self.entityTypeEmb[ner2id[node_data['ent_type']]]
            else:
                pass

            node_features[i-1] = node_feature

        edge_features = torch.zeros((len(sample['graphData']['edges']),self.edge_in_dim)).to(device)
        for i,edge_data_lst in enumerate(sample['graphData']['edges']):
            _,_,edge_data = edge_data_lst
            edge_features[i] = self.relEmb[edge_data['edge_id']]

        return node_features, edge_features

    
    def forward(self,g,node_features,edge_features,head_ent_nodes,tail_ent_nodes):
        node_h = self.gnn(g,node_features,edge_features)
        head_ent_h = node_h[head_ent_nodes]
        tail_ent_h = node_h[tail_ent_nodes]

        out = self.pred_fc(torch.cat([head_ent_h,tail_ent_h],dim=2))

        return out
