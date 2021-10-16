from re import S
import dgl
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.modules import dropout
from transformers import BertModel
from gnn.rgcn2 import multiLayerRelGraphConv


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
        self.bert = BertModel.from_pretrained("../pretrained_embed/bert-base-uncased")

        if config.fix_embed_weight:
            for p in self.bert.parameters():
                p.requires_grad = False
        else:
            for p in self.bert.parameters():
                p.requires_grad = True
            
        self.node_in_dim = self.bert.config.hidden_size
        if config.use_ner_feature:
            self.node_in_dim += config.node_ner_emb_dim
            self.nerEmb = nn.Embedding(NER_NUM + 1,config.node_ner_emb_dim,padding_idx=0)
            self.weight_init(self.nerEmb)
        if config.use_attr_feature:
            self.node_in_dim += config.node_attr_emb_dim
            self.attrEmb = nn.Embedding(NODE_ATTR_NUM,config.node_attr_emb_dim)
            self.weight_init(self.attrEmb)
        self.config = config

       
        # GNNs
        # if config.gnn == 'rgat':
        #     self.gnn = multiLayerRGAT(self.node_in_dim,config.node_dim,config.node_out_dim,config.edge_type_emb_dim,config.edge_dim,config.M,config.K,config.L,activation=make_activation(config.gnn_activation),dropout=config.dropout)
        #     self.relEmb = nn.Embedding(REL_TYPE_NUM,config.edge_type_emb_dim)
        #     for p in self.relEmb.parameters():
        #         if p.dim() >= 2:
        #             torch.nn.init.xavier_normal_(p)
        #         else:
        #             torch.nn.init.normal_(p)
        # if config.gnn == 'rgcn':
        #     self.gnn = multiLayerRGCN(self.node_in_dim,config.node_dim,config.node_out_dim,REL_TYPE_NUM,config.L,activation=make_activation(config.gnn_activation),dropout=config.dropout)
        
        if config.gnn == 'rgcn2':
            rel_names = ['loc','time','ins','mod','prep','op','ARG0','ARG1','ARG2','ARG3','ARG4','MENTION-MENTION',"DOC-MENTION","MENTION-INTER-MENTION",'others']
            # rel_names = ['MENTION-MENTION',"DOC-MENTION","MENTION-INTER-MENTION"]
            # rel_names = ['MENTION-MENTION']
            # self.gnn = multiLayerRelGraphConv(config.L,self.node_in_dim,config.node_dim,config.node_out_dim,rel_names,len(rel_names), \
            #                                   activation=make_activation(config.gnn_activation),drop_out=config.dropout)
            self.gnn = multiLayerRelGraphConv(config.L,self.node_in_dim,self.node_in_dim,self.node_in_dim,rel_names,len(rel_names), \
                                              activation=make_activation(config.gnn_activation),drop_out=config.dropout)


        # for p in self.gnn.parameters():
        #     if p.dim() >= 2:
        #         torch.nn.init.xavier_normal_(p)
        #     else:
        #         torch.nn.init.normal_(p)

        # Prediction Layer
        self.entity_embed_dim = self.node_in_dim * 2
        if config.use_edge_path:
            self.entity_embed_dim += config.node_dim
            from data_bert import etype_descriptor
            self.edge_embedding = nn.Embedding(len(etype_descriptor) + 1,config.edge_emb_dim,padding_idx=0)
            self.lstm = nn.LSTM(input_size=config.edge_emb_dim,hidden_size=config.node_dim,num_layers=config.L,batch_first=True,dropout=config.dropout)
            self.weight_init(self.edge_embedding)
            self.weight_init(self.lstm)
            # self.edge_path_fc = nn.Sequential(nn.Linear(4*config.node_dim,2*config.node_dim),
            #                          self.pred_activation,
            #                          nn.Dropout(config.dropout),
            #                          nn.Linear(2*config.node_dim,OUTPUT_NUM)
            #                         )

        if config.use_doc_feature:
            self.doc_feature_fc = nn.Sequential(nn.Linear(self.bert.config.hidden_size,self.node_in_dim),
                                                self.pred_activation,
                                                nn.Dropout(config.dropout)
                                                )
            self.weight_init(self.doc_feature_fc)

        
        self.pred_fc = nn.Sequential(nn.Linear(4*self.entity_embed_dim,2*self.entity_embed_dim),
                                     self.pred_activation,
                                     nn.Dropout(config.dropout),
                                     nn.Linear(2*self.entity_embed_dim,OUTPUT_NUM)
                                    )
        self.weight_init(self.pred_fc)

    def weight_init(self,module):
        for p in module.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.normal_(p)
        
    
    def forward(self,batch_data):
        device = self.config.device
        token_id = batch_data['token_id'].to(device) # [bsz,doc_len,feature_dim]
        token_feature, doc_cls_feature = self.bert(input_ids=token_id,attention_mask=token_id!=0)
        bsz = token_id.shape[0]

        node_span_pos = batch_data['node_span_pos'].to(device) #[bsz, max_node_num, max_span_len]
        node_span_mask = batch_data['node_span_mask'].to(device) #[bsz, max_node_num, max_span_len]

        node_span_token_features = None
        for i in range(bsz):
            features = (token_feature[i][node_span_pos[i]] * node_span_mask[i].unsqueeze(-1)).unsqueeze(0) #[max_node_num, max_span_len, feature_dim]
            node_span_token_features = torch.cat([node_span_token_features,features],dim=0) if node_span_token_features != None else features

        
        if self.config.node_span_pool_method == 'avg':
            node_span_feature = node_span_token_features.sum(dim=2)
            node_span_len = node_span_mask.sum(dim=2,keepdim=True)
            node_span_feature = node_span_feature / (node_span_len + 1e-1) # [bsz, max_node_num, feature_dim]

        node_feature = node_span_feature

        if self.config.use_doc_feature:
            node_feature += doc_cls_feature.unsqueeze(1)
            doc_cls_feature = self.doc_feature_fc(doc_cls_feature) # [bsz,node_in_dim]

        node_ner_id = batch_data['node_ner'].to(device)
        if self.config.use_ner_feature:
            node_ner_feature = self.nerEmb(node_ner_id)
            node_feature = torch.cat([node_feature, node_ner_feature],dim=-1)

        node_mask = batch_data['node_mask'].to(device)
        node_feature = node_feature * node_mask.unsqueeze(-1) #[bsz,max_node_num,feature_dim]

        batched_graph_lst = [item.to(torch.device(device)) for item in batch_data['graph']]
        # out_ent_features = torch.zeros(ent2MentionId.shape[0],ent2MentionId.shape[1],ent2MentionId.shape[2],self.entity_embed_dim,dtype=torch.float,device=device)
        batched_out = torch.zeros(batch_data['ent_pair'].shape[0],batch_data['ent_pair'].shape[1],OUTPUT_NUM,device=device)
        for i in range(bsz):
            ent2MentionId = batch_data['ent2MentionId'][i].to(device) #[max_ent_num,max_mention_nums]
            ent2MentionId_mask = batch_data['ent2MentionId_mask'][i].to(device) #[max_ent_num,max_mention_nums]
            ent_pairs = batch_data['ent_pair'][i].to(device) #[max_pair_num,2]

            graph = batched_graph_lst[i]
            # print(graph.nodes)
            node_select_idx = (node_mask[i] != 0).nonzero().squeeze(-1)
            # print(node_select_idx)
            select_node_feature = node_feature[i][node_select_idx]
            node_num = select_node_feature.shape[0]
            ment_num = ent2MentionId_mask[i].sum()

            node_in_feature = select_node_feature
            node_in_feature[ment_num] = doc_cls_feature[i] # initialize the doc_feature

            out_node_feature_lst,_ = self.gnn(graph,node_in_feature)
            out_node_feature = torch.cat([out_node_feature_lst[0],out_node_feature_lst[-1]],dim=-1) #[node_num, hid_dim]
            out_node_feature = torch.cat([torch.zeros(1,out_node_feature.shape[-1],device=device),out_node_feature],dim=0) #[node_num+1,hid_dim] for PADDING

            out_ent_feature = out_node_feature[ent2MentionId] * ent2MentionId_mask.unsqueeze(-1) #[max_ent_num,max_mention_nums,hid_dim]
            if self.config.mention_pool_method == "avg":
                out_ent_feature = out_ent_feature.sum(1) / (ent2MentionId_mask.sum(1,keepdim=True) + 1e-7) # [max_ent_num, hid_dim]

            h_feature = out_ent_feature[ent_pairs[:,0]] #[max_pair_num,dim]
            t_feature = out_ent_feature[ent_pairs[:,1]] #[max_pair_num,dim]

            if self.config.use_edge_path:
                edge_path = batch_data['edge_path'][i].to(device) # [max_pair_num, max_edge_path_num,2,max_edge_path_len,3]
                pair_num, max_edge_path_num, _, max_edge_path_len,_ = edge_path.shape
                edge_path = edge_path.reshape(-1,edge_path.shape[-2],3) # [max_pair_num * max_edge_path_num * 2,max_edge_path_len,3]
                edge_path_type_embed = self.edge_embedding(edge_path[:,:,1]) # [max_pair_num * max_edge_path_num * 2,max_edge_path_len,embed_dim]
                # edge_path_h_embed = out_node_feature[edge_path[:,:,0]]
                # edge_path_t_embed = out_node_feature[edge_path[:,:,2]]
                # edge_path_embed = torch.cat([edge_path_h_embed-edge_path_t_embed,edge_path_type_embed],dim=-1) # [max_pair_num * max_edge_path_num * 2,max_edge_path_len,embed_dim]
                edge_path_embed = edge_path_type_embed

                _,(h_n,_) = self.lstm(edge_path_embed)
                edge_path_info = h_n[-1] # only use last hidden states [max_pair_num * max_edge_path_num * 2,hidden_dim]
                edge_path_info = edge_path_info.reshape(pair_num, max_edge_path_num, 2, edge_path_info.shape[-1])
                edge_path_info = edge_path_info.sum(1) #[max_pair_num,2,hidden_dim]

                edge_path_info_h = torch.cat([edge_path_info[:,0,:],torch.zeros(h_feature.shape[0]-pair_num,edge_path_info.shape[-1],device=device)],dim=0) #[max_pair_num,hid_dim]
                edge_path_info_t = torch.cat([edge_path_info[:,1,:],torch.zeros(t_feature.shape[0]-pair_num,edge_path_info.shape[-1],device=device)],dim=0)#[max_pair_num,hid_dim]

                h_feature = torch.cat([h_feature,edge_path_info_h],dim=-1)
                t_feature = torch.cat([t_feature,edge_path_info_t],dim=-1)

            out = self.pred_fc(torch.cat([h_feature,t_feature,torch.abs(h_feature-t_feature), h_feature * t_feature],dim=-1))

            batched_out[i] = out

        return batched_out
        
        


        
        








