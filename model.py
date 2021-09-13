import dgl
from dgl.batch import batch, batch_hetero
from networkx.algorithms.cuts import edge_expansion
from torch._C import device
import torch.nn.functional as F
import copy
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils import data
from data import ner2id
from fastNLP.embeddings import BertEmbedding, ElmoEmbedding
from gnn.rgat import multiLayerRGAT
from gnn.rgcn import multiLayerRGCN


NER_NUM = 7
NODE_ATTR_NUM = 3
REL_TYPE_NUM = 15
OUTPUT_NUM = 97



def make_activation(activation_name):
    if activation_name=='relu':
        activation = nn.ReLU()
    if activation_name=='tanh':
        activation = nn.Tanh()
    if activation_name=='leacky relu':
        activation = nn.LeakyReLU()
    if activation_name=='gelu':
        activation = nn.GELU()
    if activation_name=='sigmoid':
        activation = nn.Sigmoid()

    return activation


class finalModel(nn.Module):
    def __init__(self,vocab,config):
        super(finalModel,self).__init__()

        # Activation Func
        self.pred_activation = make_activation(config.pred_activation)

        # Embedding Layer: Convert word id to word feature representation
        if config.embed_type=='bert-base':
            self.embed = BertEmbedding(vocab, model_dir_or_name=config.embed_pth,auto_truncate=True,layers='-1', pool_method=config.embed_pool_method)
        if config.embed_type=='Elmo':
            self.embed = ElmoEmbedding(vocab, model_dir_or_name=config.embed_pth, requires_grad=True, layers='mix')

        if config.fix_embed_weight:
            for p in self.embed.parameters():
                p.requires_grad = False
        else:
            for p in self.embed.parameters():
                p.requires_grad = True
            
        self.node_in_dim = self.embed.embedding_dim
        if config.use_ner_feature:
            self.node_in_dim += config.node_ner_emb_dim
            self.nerEmb = nn.Embedding(NER_NUM,config.node_ner_emb_dim)
            for p in self.nerEmb.parameters():
                if p.dim() >= 2:
                    torch.nn.init.xavier_normal_(p)
                else:
                    torch.nn.init.normal_(p)
        if config.use_attr_feature:
            self.node_in_dim += config.node_attr_emb_dim
            self.attrEmb = nn.Embedding(NODE_ATTR_NUM,config.node_attr_emb_dim)
            for p in self.attrEmb.parameters():
                if p.dim() >= 2:
                    torch.nn.init.xavier_normal_(p)
                else:
                    torch.nn.init.normal_(p)
        self.config = config

        # GNNs
        if config.gnn == 'rgat':
            self.gnn = multiLayerRGAT(self.node_in_dim,config.node_dim,config.node_out_dim,config.edge_in_dim,config.edge_dim,config.M,config.K,config.L,activation=config.gnn_activation)
            self.relEmb = nn.Embedding(REL_TYPE_NUM,config.edge_type_emb_dim)
            for p in self.relEmb.parameters():
                if p.dim() >= 2:
                    torch.nn.init.xavier_normal_(p)
                else:
                    torch.nn.init.normal_(p)
        if config.gnn == 'rgcn':
            self.gnn = multiLayerRGCN(self.node_in_dim,config.node_dim,config.node_out_dim,REL_TYPE_NUM,config.L,activation=config.gnn_activation)
        
        for p in self.gnn.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.normal_(p)

        # Prediction Layer
        self.pred_fc = nn.Sequential(nn.Linear(4*self.node_out_dim,2*self.node_out_dim),
                                     self.pred_activation,
                                     nn.Linear(2*self.node_out_dim,OUTPUT_NUM)
                                    )
        for p in self.pred_fc.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.normal_(p)
    
    def forward(self,batch_data):
        device = self.config.device
        token_id = batch_data['token_id'].to(device) # [bsz,doc_len]
        token_feature = self.embed(token_id) # [bsz,doc_len,token_emb_dim]
        bsz = token_id.shape[0]

        batched_node_in_feature = None
        batched_edge_in_feature = None
        node_offset = 0
        ent_pair_pos = batch_data['ent_pair'].to(device) #[bsz,max_pair_num,2]
        batch_data['graph'] = [batch_data['graph'][i].to(device) for i in range(bsz)]
        for i,g in enumerate(batch_data['graph']):    
          
            if self.config.node_span_pool_method == 'avg':
                node_span_mask = g.ndata['span_mask'] # [node_num,doc_len]
                node_span_feature = node_span_mask.unsqueeze(-1) * (token_feature[i].unsqueeze(0).expand(g.num_nodes(),-1,-1)) # [node_num, doc_len, feature_dim]
                node_span_feature = node_span_feature.sum(1) #[node_num,feature_dim]
                node_span_len = node_span_mask.sum(1,keepdim=True)
                node_span_feature = torch.div(node_span_feature, node_span_len + 1e-6) # avoid divide zero
            
            node_in_feature = node_span_feature     # [bsz,node_num,node_span_feature_dim]

            if self.config.use_ner_feature:
                node_ner_feature = self.nerEmb(g.ndata['ner_id'])
                node_in_feature = torch.cat([node_in_feature,node_ner_feature],dim=1)
            if self.config.use_attr_feature:
                node_attr_feature = self.attrEmb(g.ndata['attr_id'])
                node_in_feature = torch.cat([node_in_feature,node_attr_feature],dim=1)

            edge_in_feature = self.relEmb(g.edata['edge_id'])
            
            batched_node_in_feature = torch.cat([batched_node_in_feature,node_in_feature],dim=0) if batched_node_in_feature != None else node_in_feature
            batched_edge_in_feature = torch.cat([batched_edge_in_feature,edge_in_feature],dim=0) if batched_edge_in_feature != None else edge_in_feature

            ent_pair_pos[i] += node_offset
            node_offset += g.num_nodes()
        batched_g = dgl.batch(batch_data['graph'])
        node_out_feature = self.gnn(batched_g,batched_node_in_feature,batched_edge_in_feature)


        # Get Representation of Each ENtity. Predict pair-wise relation
        h_feature = node_out_feature[ent_pair_pos[:,:,0]]
        t_feature = node_out_feature[ent_pair_pos[:,:,1]]

        out = self.pred_fc(torch.cat([h_feature,t_feature,torch.abs(h_feature-t_feature), h_feature * t_feature],dim=-1))

        return out
        
        
            

        
        
        
        




class debugGNN(nn.Module):
    def __init__(self,node_in_dim,node_dim):
        ## 
        super(debugGNN,self).__init__()

        # Hyperparameter
        self.node_dim = node_dim
        self.node_in_dim = node_in_dim

        self.node_fc = nn.Sequential(nn.Linear(node_in_dim,node_dim),nn.ReLU())

    def message_func(self,edges):
        return {'h_j': edges.dst['h']}
    
    def reduce_func(self,nodes):
        h_j = nodes.mailbox['h_j']  #[Node_num,N_i,node_dim]
        N_i = h_j.shape[1]
        h = torch.sum(h_j,dim=1) / N_i

        return {'h':h}

    def forward(self,g,node_features,edge_features):
        g.ndata['h'] = self.node_fc(node_features)

        g.update_all(self.message_func,self.reduce_func)

        # out_features = g.ndata.pop('h')
        out_features = g.ndata['h']

        return out_features








class debugModel(nn.Module):
    def __init__(self,embed,node_in_dim,node_dim,node_out_dim,edge_in_dim,edge_dim,M,K,L):
        super(debugModel,self).__init__()

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
        self.gnn = debugGNN(node_in_dim,node_dim)
        self.entityTypeEmb = Parameter(torch.randn((node_in_dim,node_in_dim)))
        self.relEmb = Parameter(torch.randn((REL_TYPE_NUM,REL_TYPE_EMB_DIM)))

        # Prediction Layer
        self.pred_fc = nn.Linear(2*node_dim,OUTPUT_NUM)

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

