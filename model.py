import dgl
from dgl.batch import batch
import torch.nn.functional as F
import copy
import torch
import torch.nn as nn
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
            self.nerEmb = nn.Embedding(NER_NUM + 1,config.node_ner_emb_dim)
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
            self.gnn = multiLayerRGAT(self.node_in_dim,config.node_dim,config.node_out_dim,config.edge_type_emb_dim,config.edge_dim,config.M,config.K,config.L,activation=make_activation(config.gnn_activation))
            self.relEmb = nn.Embedding(REL_TYPE_NUM,config.edge_type_emb_dim)
            for p in self.relEmb.parameters():
                if p.dim() >= 2:
                    torch.nn.init.xavier_normal_(p)
                else:
                    torch.nn.init.normal_(p)
        if config.gnn == 'rgcn':
            self.gnn = multiLayerRGCN(self.node_in_dim,config.node_dim,config.node_out_dim,REL_TYPE_NUM,config.L,activation=make_activation(config.gnn_activation))
        
        for p in self.gnn.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.normal_(p)

        # Prediction Layer
        self.pred_fc = nn.Sequential(nn.Linear(4*config.node_out_dim,2*config.node_out_dim),
                                     self.pred_activation,
                                     nn.Linear(2*config.node_out_dim,OUTPUT_NUM)
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
                node_span_mask = g.ndata['span_mask'] # [node_num,max_span_len]
                node_span_pos = g.ndata['span_pos'] # [node_num,max_span_len]
                node_span_feature = token_feature[i][node_span_pos] # [node_num,max_span_len,feature_dim]
                node_span_feature = node_span_feature * node_span_mask.unsqueeze(-1)
                node_span_feature = node_span_feature.sum(1)
                node_span_len = node_span_mask.sum(1,keepdim=True)
                node_span_feature = torch.div(node_span_feature, node_span_len + 1e-7)
                # node_span_feature = node_span_mask.unsqueeze(-1) * (token_feature[i].unsqueeze(0).expand(g.num_nodes(),-1,-1)) # [node_num, doc_len, feature_dim]
                # node_span_feature = node_span_feature.sum(1) #[node_num,feature_dim]
                # node_span_len = node_span_mask.sum(1,keepdim=True)
                # node_span_feature = torch.div(node_span_feature, node_span_len + 1e-6) # avoid divide zero
            
            node_in_feature = node_span_feature     # [bsz,node_num,node_span_feature_dim]

            if self.config.use_ner_feature:
                node_ner_feature = self.nerEmb(g.ndata['ner_id'])
                node_in_feature = torch.cat([node_in_feature,node_ner_feature],dim=1)
            if self.config.use_attr_feature:
                node_attr_feature = self.attrEmb(g.ndata['attr_id'])
                node_in_feature = torch.cat([node_in_feature,node_attr_feature],dim=1)

            if self.config.gnn == 'rgat':
                edge_in_feature = self.relEmb(g.edata['edge_id'])
                batched_edge_in_feature = torch.cat([batched_edge_in_feature,edge_in_feature],dim=0) if batched_edge_in_feature != None else edge_in_feature
            
            batched_node_in_feature = torch.cat([batched_node_in_feature,node_in_feature],dim=0) if batched_node_in_feature != None else node_in_feature

            ent_pair_pos[i] += node_offset
            node_offset += g.num_nodes()
        batched_g = dgl.batch(batch_data['graph'])

        if self.config.gnn == 'rgat':
            node_out_feature = self.gnn(batched_g,batched_node_in_feature,batched_edge_in_feature)
        
        if self.config.gnn == 'rgcn':
            node_out_feature = self.gnn(batched_g,batched_node_in_feature)

        # Get Representation of Each ENtity. Predict pair-wise relation
        h_feature = node_out_feature[ent_pair_pos[:,:,0]]
        t_feature = node_out_feature[ent_pair_pos[:,:,1]]

        out = self.pred_fc(torch.cat([h_feature,t_feature,torch.abs(h_feature-t_feature), h_feature * t_feature],dim=-1))

        return out
        
        
            

        
        








