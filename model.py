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
    def __init__(self,vocab,config):
        super(finalModel,self).__init__()

        # Model HyperParameters
        self.attr_emb_dim = config.node_ner_emb_dim
        self.node_type_emb_dim = config.node_attr_emb_dim

        # Activation Func
        self.pred_activation = make_activation(config.pred_activatino)

        # Embedding Layer: Convert word id to word feature representation
        if config.embed_type=='bert-base':
            self.embed = BertEmbedding(vocab, model_dir_or_name=config.embed_pth,auto_truncate=True,layers='-1', pool_method=config.embed_pool_method)
        if config.embed_type=='Elmo':
            self.embed = ElmoEmbedding(vocab, model_dir_or_name=config.embed_pth, requires_grad=True, layers='mix')
        for p in self.embed.parameters():
            p.requires_grad=True
        
        self.node_in_dim = self.embed.embedding_dim
        if config.use_ner_feature:
            self.node_in_dim += config.node_ner_emb_dim
        if config.use_attr_feature:
            self.node_in_dim += config.node_attr_emb_dim
        self.node_dim = config.node_dim
        self.node_out_dim = config.node_out_dim
        self.edge_in_dim = config.edge_type_emb_dim
        self.edge_dim = config.edge_dim
        self.M = config.M
        self.K = config.K
        self.L = config.L
        self.config = config

        # GNNs
        self.gnn = multiLayerRGAT(self.node_in_dim,self.node_dim,self.node_out_dim,self.edge_in_dim,self.edge_dim,self.M,self.K,self.L)
        self.nerEmb = nn.Embedding(NER_NUM,config.node_ner_emb_dim)
        self.attrEmb = nn.Embedding(NODE_ATTR_NUM,config.node_attr_emb_dim)
        self.relEmb = nn.Embedding(REL_TYPE_NUM,config.edge_type_emb_dim)

        # Prediction Layer
        self.pred_fc = nn.Sequential(nn.Linear(4*self.node_out_dim,2*self.node_out_dim),
                                     self.pred_activation,
                                     nn.Linear(2*self.node_out_dim,OUTPUT_NUM)
                                    )
    
    def init_params(self):
        for p in self.gnn.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.normal_(p)
        torch.nn.init.xavier_normal_(self.nerEmb)
        torch.nn.init.xavier_normal_(self.attrEmb)
        torch.nn.init.xavier_normal_(self.relEmb)
        for p in self.pred_fc.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.normal_(p)

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

    
    def forward(self,batch_data):
        device = self.embed.device
        token_id = batch_data['token_id'].to(device) # [bsz,doc_len]
        token_feature = self.embed(token_id) # [bsz,doc_len,token_emb_dim]

        batched_node_in_feature = None
        batched_edge_in_feature = None
        node_offset = 0
        ent_pair_pos = batch_data['ent_pair'].to(device) #[bsz,max_pair_num,2]
        for i,g in enumerate(batch_data['graph']):    
            g = g.to(device)

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
            
            batched_node_in_feature = torch.cat([batched_node_in_feature,node_in_feature],dim=0) if batched_node_in_feature is None else node_in_feature
            batched_edge_in_feature = torch.cat([batched_edge_in_feature,edge_in_feature],dim=0) if batched_edge_in_feature is None else edge_in_feature

            ent_pair_pos[i] += node_offset
            node_offset += g.num_nodes()
        batched_g = dgl.batch(batch_data['graph'])
        node_out_feature = self.gnn(batched_g,batched_node_in_feature,batched_edge_in_feature)
        h_feature = node_out_feature[ent_pair_pos[:,:,0]]
        t_feature = node_out_feature[ent_pair_pos[:,:,1]]

        out = self.pred_fc(torch.cat(h_feature,t_feature,torch.abs(h_feature-t_feature)), h_feature * t_feature)

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

