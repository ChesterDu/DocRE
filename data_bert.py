from dgl import batch
from fastNLP.core.vocabulary import Vocabulary
import networkx as nx
import json
from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor
import torch
import pickle
import dgl
import copy
import random
import numpy as np
import os
from collections import defaultdict
from transformers import BertTokenizer

with open("../DocRED/rel2id.json","r") as fp:
    rel2id = json.load(fp)
with open("../DocRED/ner2id.json","r") as fp:
    ner2id = json.load(fp)

attr2id = {'entity':0,"mention":1,"amr":2}

ntype_descriptor = ['nodes']
etype_descriptor = ['loc','time','ins','mod','prep','op','ARG0','ARG1','ARG2','ARG3','ARG4','MENTION-MENTION',"DOC-MENTION","MENTION-INTER-MENTION",'others']
## Function that convert edge type string to Id
def get_edge_idx(edge_type_str):    
        if edge_type_str in ['location', 'destination', 'path']:
            return 0
        elif edge_type_str in ['year', 'time', 'duration', 'decade', 'weekday']:
            return 1
        elif edge_type_str in ['instrument', 'manner', 'poss', 'topic', 'medium', 'duration']:
            return 2
        elif edge_type_str in ['mod']:
            return 3
        elif edge_type_str.startswith('prep-'):
            return 4
        elif edge_type_str.startswith('op') and edge_type_str[-1].isdigit():
            return 5
        elif edge_type_str == 'ARG0':
            return 6
        elif edge_type_str == 'ARG1':
            return 7
        elif edge_type_str == 'ARG2':
            return 8
        elif edge_type_str == 'ARG3':
            return 9
        elif edge_type_str == 'ARG4':
            return 10
        elif edge_type_str == "MENTION-MENTION":
            return 11
        elif edge_type_str == "DOC-MENTION":
            return 12
        elif edge_type_str == "MENTION-INTER-MENTION":
            return 13
        else:
            return 14


def build_vocab(train_data_pth,test_data_pth,dev_data_pth):
    vocab = Vocabulary(min_freq=10)
    def add_words_to_vocab(data_pth,vocab):
        with open(data_pth,'r') as fp:
            samples = json.load(fp)
        for sample in samples:
            for amr_graph in sample['amrGraphs']:
                for u,edge_type,v in amr_graph['edges']:
                    vocab.add_word(edge_type)
        return vocab
    
    vocab = add_words_to_vocab(train_data_pth,vocab)
    vocab = add_words_to_vocab(test_data_pth,vocab)
    vocab = add_words_to_vocab(dev_data_pth,vocab)

    return vocab

class BertGraphDataset(torch.utils.data.Dataset):
    def __init__(self,config,processed_data_pth,raw_data_pth,vocab,ignore_label_idx=-1,split='train',fact_in_train=set([])):
        super(BertGraphDataset,self).__init__()

        random.seed(config.seed)

        with open(raw_data_pth,'r') as fp:
            self.samples = json.load(fp)
            if config.debug:
                self.samples = self.samples[:20]
        
        # self.create_amr_graph_alighments()
        # print("AMR Graph Alignments Completed!")
        
        self.vocab = vocab
        self.max_subword_token_len = config.max_token_len
        self.max_sen_num = max([len(sample['sents']) for sample in self.samples])
        self.split = split
        # self.get_token_ids()

        self.max_ent_num = max([len(sample['vertexSet']) for sample in self.samples])
        # self.max_pair_num = self.max_ent_num * (self.max_ent_num - 1)
        self.max_pair_num = max([len(sample['labels']) for sample in self.samples])
        self.ignore_label_idx = ignore_label_idx

        self.naPairs_alpha = config.naPairs_alpha
        
        self.fact_in_train = fact_in_train
        self.use_amr_graph = config.use_amr_graph

        self.bert_utils = BertUtils('bert-base-uncased',model_path='../pretrained_embed/bert-base-uncased')
        self.config = config
        # self.create_labels()

        # if os.path.exists(processed_data_pth) and (not config.debug):
        #     with open(processed_data_pth,'rb') as fp:
        #         data = pickle.load(fp)
        #         self.samples = data['samples']
        #         self.fact_in_train = data['fact_in_train']
        #         self.manipulate_naPairs_num()
        # else:
        #     self.process_data()
        #     if not config.debug:
        #         with open(processed_data_pth,'wb') as fp:
        #             pickle.dump({"samples":self.samples,"fact_in_train":fact_in_train},fp)
        self.process_data()
        self.print_stat()


    def __getitem__(self,index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)
    
    def print_stat(self):
        print("==============Data Statistic===================")
        print("Split: {} || Sample Num: {} || Max Token Num: {} || Max Ent Num: {} || Max Pair Num: {}".format(self.split,len(self.samples),self.max_subword_token_len,self.max_ent_num,self.max_pair_num))

    def manipulate_naPairs_num(self):
        if self.split == 'train':
            for doc_id,doc in enumerate(self.samples):
                h_t_pair2label = self.samples[doc_id]['posPairs2label']
                self.samples[doc_id]['naPairs_num'] = int(max([10,self.naPairs_alpha * len(h_t_pair2label)]))
        else:
            for doc_id,doc in enumerate(self.samples):
                self.samples[doc_id]['naPairs_num'] = 0
            


    def process_data(self):
        for doc_id,doc in enumerate(self.samples):
            ## Add * token before and after mention tokens
            sen_id2mentions = defaultdict(list)
            vertexSet = copy.deepcopy(doc['vertexSet'])
            for ent_id,mentions in enumerate(vertexSet):
                for men_id,mention in enumerate(mentions):
                    mention['ent_id'] = ent_id
                    sen_id2mentions[mention['sent_id']].append(mention)

            doc['vertexSet_new'] = [[] for i in range(len(doc['vertexSet']))]
            doc['sents_new'] = copy.deepcopy(doc['sents'])
            for sen_id,mentions in sen_id2mentions.items():
                sorted_mentions = sorted(mentions,key=lambda mention:mention['pos'][0])
                offset = 0
                for mention in sorted_mentions:
                    doc['sents_new'][sen_id].insert(mention['pos'][0]+offset,"*")
                    doc['sents_new'][sen_id].insert(mention['pos'][1]+1+offset,"*")
                    new_start_pos = mention['pos'][0] + offset
                    new_end_pos = mention['pos'][1] + 2 + offset
                    mention['pos_new'] = [new_start_pos,new_end_pos]

                    offset += 2
                    
                    doc['vertexSet_new'][mention['ent_id']].append(mention)

            ## convert word to ids
            words = []
            sen_start_pos_lst = [0]
            for i,sen in enumerate(doc['sents_new']):
            # for i,sen in enumerate(doc['sents']):
                sen_start_pos = sen_start_pos_lst[-1] + len(sen)
                sen_start_pos_lst.append(sen_start_pos)
                for word in sen:
                    words.append(word)
                    
            bert_subword_token_ids, bert_subword_starts,bert_subwords = self.bert_utils.subword_tokenize_to_ids(words)
            sen_start_pos_lst = sen_start_pos_lst[:-1]
            
            doc['tokenIds'] = bert_subword_token_ids[0]

            ## Generate position mapping betweeen Mentions and Subword_ids/ Mention - AMR graph Mapping
            mention_pos_lst = []
            global_id = 0
            entId2mentionGlobalId = []
            mentionGlobalId2entId = []
            senId2mentionGlobalId = defaultdict(list)
            mentionGlobalId2amrNodeAlignment = []
            mention_nerId_lst = []
            amr_graphs = doc['amrGraphs']
            # for ent_id,mentions in enumerate(doc['vertexSet_new']):
            for ent_id,mentions in enumerate(doc['vertexSet_new']):
                mentionGlobalIds = []
                for men_id,mention in enumerate(mentions):
                    ### Generate Position Mapping between Mentions and Subword_ids
                    men_start_pos = mention['pos_new'][0] + sen_start_pos_lst[mention['sent_id']]
                    men_subword_start_pos = bert_subword_starts[men_start_pos]
                    men_end_pos = mention['pos_new'][1] + sen_start_pos_lst[mention['sent_id']]
                    men_subword_end_pos = bert_subword_starts[men_end_pos]
                    mention_pos_lst.append(list(range(men_subword_start_pos,men_subword_end_pos)))
                    mentionGlobalIds.append(global_id)
                    mention_nerId_lst.append(ner2id[mention['type']] + 1)
                    senId2mentionGlobalId[mention['sent_id']].append(global_id)

                    ### Generate Mention - AMR graph Mapping
                    amr_graph = amr_graphs[mention['sent_id']]
                    mention_pos = mention['pos']
                    mention_end_pos = mention_pos[1] - 1
                    #### Find Alignment follow the rule
                    match_flag = False
                    for amr_node in amr_graph['nodes']:
                        try:
                            amr_span_pos = amr_graph['alignments'][amr_node]
                        except KeyError:
                            continue
                        if type(amr_span_pos) != list:
                            amr_span_pos = [amr_span_pos]
                        
                        if (mention_end_pos >= (amr_span_pos[0] - 1)) and (mention_end_pos <= (amr_span_pos[-1] - 1)):
                            mentionGlobalId2amrNodeAlignment.append((mention['sent_id'],amr_node))
                            match_flag = True
                            break
                    
                    #### If can not find direct mapping, then adapt the rule
                    if match_flag == False:
                        matched_amr_node = None
                        minimum_span_distance = 10000
                        for amr_node in amr_graph['nodes']:
                            try:
                                amr_span_pos = amr_graph['alignments'][amr_node]
                            except KeyError:
                                continue
                            if type(amr_span_pos) != list:
                                amr_span_pos = [amr_span_pos]

                            span_dist = abs(mention_end_pos - (amr_span_pos[0] - 1)) + \
                                        abs(mention_end_pos - (amr_span_pos[-1] - 1))
                            
                            if span_dist < minimum_span_distance:
                                minimum_span_distance = span_dist
                                matched_amr_node = amr_node
                        
                        mentionGlobalId2amrNodeAlignment.append((mention['sent_id'],matched_amr_node))
                    mentionGlobalId2entId.append(ent_id)

                    global_id += 1
                entId2mentionGlobalId.append(mentionGlobalIds)
            
            doc['mentionPos'] = mention_pos_lst
            doc['mentionNer'] = mention_nerId_lst
            doc['ent2MentionId'] = entId2mentionGlobalId


             ## Generate Graph Topology
            d = defaultdict(list)
            mention_num = sum([len(mentions) for mentions in entId2mentionGlobalId])
            doc_node_id = mention_num
            for ent_id, mentions in enumerate(entId2mentionGlobalId):
                for ment_id in mentions:
                    d[('nodes','DOC-MENTION','nodes')].append((doc_node_id,ment_id))
                    for ment_id_j in mentions:
                        if ment_id_j == ment_id:
                            continue
                        d[('nodes','MENTION-MENTION','nodes')].append((ment_id,ment_id_j))
                
            for sent_id, mentions in senId2mentionGlobalId.items():
                for ment_id in mentions:
                    for ment_id_j in mentions:
                        if ment_id_j == ment_id or (ment_id,ment_id_j) in d[('nodes','MENTION-MENTION','nodes')]:
                            continue
                        d[('nodes',"MENTION-INTER-MENTION","nodes")].append((ment_id,ment_id_j))

            doc['nodeSpanPos'] = mention_pos_lst + [[]]
            if self.config.use_amr_graph:
                amrNode2nodeId = {}
                amr_node_id = doc_node_id + 1
                amr_node_span_pos_lst = []
                amrNodeAlignment2mentionGlobalId = {v:i for i,v in enumerate(mentionGlobalId2amrNodeAlignment)}

                amr_graphs = doc['amrGraphs']
                for sent_id,amr_graph in enumerate(amr_graphs):
                    for u,amr_rel,v in amr_graph['edges']:
                        if ((sent_id,u) not in amrNode2nodeId) and ((sent_id,str(u)) not in amrNodeAlignment2mentionGlobalId):
                            amrNode2nodeId[(sent_id,u)] = amr_node_id
                            try:
                                amr_node_span_pos = amr_graph['alignments'][str(u)]
                            except KeyError:
                                amr_node_span_pos = []
                            
                            if amr_node_span_pos == []:
                                amr_node_span_pos_lst.append(amr_node_span_pos)
                                amr_node_id += 1
                            
                            else:
                                if type(amr_node_span_pos) != list:
                                    amr_node_span_pos = [amr_node_span_pos]
                                
                                amr_node_start_pos = amr_node_span_pos[0]
                                if amr_node_start_pos < 0:
                                    amr_node_start_pos = len(doc['sents'][sen_id]) + amr_node_start_pos + sen_start_pos_lst[sen_id]
                                else:
                                    amr_node_start_pos = amr_node_start_pos + sen_start_pos_lst[sen_id] - 1

                                amr_node_end_pos = amr_node_span_pos[-1]
                                if amr_node_end_pos < 0:
                                    amr_node_end_pos = len(doc['sents'][sen_id]) + amr_node_start_pos + sen_start_pos_lst[sen_id] +1
                                else:
                                    amr_node_end_pos = amr_node_end_pos + sen_start_pos_lst[sen_id]

                                try:
                                    amr_node_start_pos = bert_subword_starts[amr_node_start_pos]
                                except:
                                    amr_node_start_pos = self.max_subword_token_len - 1
                                
                                try:
                                    amr_node_end_pos = bert_subword_starts[amr_node_end_pos]
                                except:
                                    amr_node_end_pos = self.max_subword_token_len

                                amr_node_span_pos_lst.append(list(range(amr_node_start_pos,amr_node_end_pos)))
                                amr_node_id += 1
                        
                        if ((sent_id,v) not in amrNode2nodeId) and ((sent_id,str(v)) not in amrNodeAlignment2mentionGlobalId):
                            amrNode2nodeId[(sent_id,v)] = amr_node_id
                            try:
                                amr_node_span_pos = amr_graph['alignments'][str(v)]
                            except KeyError:
                                amr_node_span_pos = []
                            
                            if amr_node_span_pos == []:
                                amr_node_span_pos_lst.append(amr_node_span_pos)
                                amr_node_id += 1
                            
                            else:
                                if type(amr_node_span_pos) != list:
                                    amr_node_span_pos = [amr_node_span_pos]
                                
                                amr_node_start_pos = amr_node_span_pos[0]
                                if amr_node_start_pos < 0:
                                    amr_node_start_pos = len(doc['sents'][sen_id]) + amr_node_start_pos + sen_start_pos_lst[sen_id]
                                else:
                                    amr_node_start_pos = amr_node_start_pos + sen_start_pos_lst[sen_id] - 1

                                amr_node_end_pos = amr_node_span_pos[-1]
                                if amr_node_end_pos < 0:
                                    amr_node_end_pos = len(doc['sents'][sen_id]) + amr_node_start_pos + sen_start_pos_lst[sen_id] +1
                                else:
                                    amr_node_end_pos = amr_node_end_pos + sen_start_pos_lst[sen_id]

                                try:
                                    amr_node_start_pos = bert_subword_starts[amr_node_start_pos]
                                except:
                                    amr_node_start_pos = self.max_subword_token_len - 1
                                
                                try:
                                    amr_node_end_pos = bert_subword_starts[amr_node_end_pos]
                                except:
                                    amr_node_end_pos = self.max_subword_token_len

                                amr_node_span_pos_lst.append(list(range(amr_node_start_pos,amr_node_end_pos)))
                                amr_node_id += 1
                        
                        def get_amr_node_id(sent_id,u):
                            try:
                                return amrNode2nodeId[(sent_id,u)]
                            except KeyError:
                                return amrNodeAlignment2mentionGlobalId[(sent_id,str(u))]
                        
                        node_id_u = get_amr_node_id(sent_id,u)
                        node_id_v = get_amr_node_id(sent_id,v)
                        etype_str = etype_descriptor[get_edge_idx(amr_rel[1:])]
                        d[('nodes',etype_str,'nodes')].append((node_id_u,node_id_v))
                doc['amrNodePos'] = amr_node_span_pos_lst
                doc['nodeSpanPos'] += doc['amrNodePos']

            num_nodes = len(doc['nodeSpanPos'])
            A = np.zeros([num_nodes,num_nodes])
            for (_,etype_str,_),edges in d.items():
                etype_id = etype_descriptor.index(etype_str)
                for u,v in edges:
                    A[u][v] = etype_id + 1
                    # A[v][u] = 1
            # doc['nodeSpanPos'] = mention_pos_lst + [[]]
            doc['graph'] = d
            doc['adjMatrix'] = A.tolist()
                    
            if self.config.use_edge_path:
                mentIdPair2amrPath = {}
                for sent_id, mentions in senId2mentionGlobalId.items():
                    g = nx.DiGraph()
                    amr_graph = amr_graphs[sent_id]
                    for u,etype,v in amr_graph['edges']:
                        g.add_edge(u,v,etype=etype)
                    
                    if g.number_of_nodes() == 0:
                        continue
                    
                    for ment_i in mentions:
                        for ment_j in mentions:
                            if ment_i == ment_j: 
                                continue
                            amr_node_i = mentionGlobalId2amrNodeAlignment[ment_i][1]
                            amr_node_j = mentionGlobalId2amrNodeAlignment[ment_j][1]

                            try:
                                lowest_common_ancestor = nx.lowest_common_ancestor(g,int(amr_node_i),int(amr_node_j))
                            except nx.NetworkXError:
                                lowest_common_ancestor = amr_graph['root']
                            try:
                                path1 = nx.shortest_path(g,source=int(lowest_common_ancestor),target=int(amr_node_i))
                            except nx.NetworkXNoPath:
                                path1 = []
                            try:
                                path2 = nx.shortest_path(g,source=int(lowest_common_ancestor),target=int(amr_node_j))
                            except nx.NetworkXNoPath:
                                path2 = []
                            def tuple_path(path_lst,g):
                                u = path_lst[0]
                                route_lst = []
                                for i in range(1,len(path_lst)):
                                    u = path_lst[i-1]
                                    v = path_lst[i]
                                    route_lst.append((u,g.edges[u,v]['etype'],v))
                                    
                                return route_lst

                            mentIdPair2amrPath[(ment_i,ment_j)] = [tuple_path(path1,g),tuple_path(path2,g),sent_id]

                doc['mentIdPair2amrPath'] = mentIdPair2amrPath

                entPair2mentIdPair = {}
                for ent_i in range(len(doc['vertexSet'])):
                    for ent_j in range(len(doc['vertexSet'])):
                        if ent_i == ent_j:
                            continue
                        entPair2mentIdPair[(ent_i,ent_j)] = []
                        ent_i_mentions = entId2mentionGlobalId[ent_i]
                        ent_j_mentions = entId2mentionGlobalId[ent_j]

                        for ent_i_mention in ent_i_mentions:
                            for ent_j_mention in ent_j_mentions:
                                if (ent_i_mention, ent_j_mention) in mentIdPair2amrPath:
                                    entPair2mentIdPair[(ent_i,ent_j)].append((ent_i_mention,ent_j_mention))
                
                doc['entPair2mentIdPair'] = entPair2mentIdPair

                entPair2amrEdgePath = {}
                for ent_pair,relate_mention_pairs in entPair2mentIdPair.items():
                    amrEdgePath = []
                    for mention_pair in relate_mention_pairs:
                        tuple_path_i,tuple_path_j,sen_id = mentIdPair2amrPath[mention_pair]
                        edge_path_i = [[get_amr_node_id(sen_id,item[0]),get_edge_idx(item[1][1:]),get_amr_node_id(sen_id,item[2])] for item in tuple_path_i]
                        edge_path_j = [[get_amr_node_id(sen_id,item[0]),get_edge_idx(item[1][1:]),get_amr_node_id(sen_id,item[2])] for item in tuple_path_j]
                        amrEdgePath.append([edge_path_i,edge_path_j])
                    entPair2amrEdgePath[ent_pair] = amrEdgePath
                
                doc['entPair2amrEdgePath'] = entPair2amrEdgePath

            ## Generate Entity Pairs and Labels

            ### Positive labels
            h_t_pair2label = {}
            # offset = mention_num
            offset = 0
            for i,label_data in enumerate(doc['labels']):
                h_idx,t_idx,r = label_data['h']+offset,label_data['t']+offset,rel2id[label_data['r']]
                pair = (h_idx,t_idx)
                if pair not in h_t_pair2label:
                    h_t_pair2label[pair] = [r]
                else:
                    h_t_pair2label[pair].append(r)

            ### Negative labels
            entNum = len(doc['vertexSet'])
            naPairs = []
            for i in range(entNum):
                for j in range(entNum):
                    if i==j:
                        continue
                    pair = (i+offset,j+offset)
                    if pair not in h_t_pair2label:
                        naPairs.append([pair[0],pair[1]])
            random.shuffle(naPairs)
            
            doc['posPairs2label'] = h_t_pair2label
            doc['naPairs'] = naPairs    
            doc['naPairs_num'] = int(max([10,self.naPairs_alpha * len(h_t_pair2label)]))
            if self.split=='dev':
                doc['naPairs_num'] = 0



def collate_fn(batch_samples):  
    batched_token_id = []
    batched_graph = []
    batched_node_pos = []
    batched_node_ner = []
    batched_node_span_mask = []
    batched_node_mask = []
    batched_adj_matrix = []
    # batched_edge_path = []
    # batched_edge_path_length = []

    batched_label = []
    batched_multi_label = []
    batched_entPair = []
    batched_titles = []
    batched_ent2MentionId = [] # [bsz, max_ent, max_mention]
    batched_ent2MentionId_mask = [] # [bsz, max_ent, max_mention]

    max_span_len = max([max([len(item) for item in sample['nodeSpanPos']]) for sample in batch_samples])
    max_node_num = max([len(sample['nodeSpanPos']) for sample in batch_samples])
    max_ent_num = max([len(sample['vertexSet']) for sample in batch_samples])
    max_ent2mention_num = max([max([len(item) for item in sample['ent2MentionId']]) for sample in batch_samples])
    for sample in batch_samples:
        batched_token_id.append(sample['tokenIds'])

        G = dgl.heterograph(sample['graph'])
        batched_graph.append(G)

        batched_titles.append(sample['title'])

        node_span_mask = []
        node_span_pos = []
        node_mask = []
        node_ner = sample['mentionNer']
        for span_pos_lst in sample['nodeSpanPos']:
            span_mask = [1] * len(span_pos_lst) + [0] * (max_span_len - len(span_pos_lst))
            span_pos_lst = span_pos_lst + [0] * (max_span_len - len(span_pos_lst))
            node_span_mask.append(span_mask)
            node_span_pos.append(span_pos_lst)
            node_mask.append(1)
        node_span_pos += [[0] * max_span_len for i in range(max_node_num - len(node_span_pos))]
        node_span_mask += [[0] * max_span_len for i in range(max_node_num - len(node_span_mask))]
        node_mask += [0] * (max_node_num - len(node_mask))
        node_ner += [0] * (max_node_num - len(node_ner))

        batched_node_pos.append(node_span_pos)
        batched_node_span_mask.append(node_span_mask)
        batched_node_mask.append(node_mask)
        batched_node_ner.append(node_ner)

        ent2MentionId = []
        ent2MentionId_mask = []
        for mentions_id in sample['ent2MentionId']:
            mentions_id_mask = [1] * len(mentions_id) + [0] * (max_ent2mention_num - len(mentions_id))
            mentions_id += [0] * (max_ent2mention_num - len(mentions_id))
            # mentions_id = [item + 1 for item in mentions_id]
            ent2MentionId_mask.append(mentions_id_mask)
            ent2MentionId.append(mentions_id)
        
        ent2MentionId += [[0] * max_ent2mention_num for i in range(max_ent_num - len(ent2MentionId))]
        ent2MentionId_mask += [[0] * max_ent2mention_num for i in range(max_ent_num - len(ent2MentionId_mask))]
        batched_ent2MentionId.append(ent2MentionId)
        batched_ent2MentionId_mask.append(ent2MentionId_mask)

       
        naPairs_num = min([len(sample['naPairs']), sample['naPairs_num']])
        if sample['naPairs_num'] == 0:
            naPairs_num = len(sample['naPairs'])
        
        naPairs = sample['naPairs'][:naPairs_num]
        posPairs2label = sample['posPairs2label']
        pairs = []

        # edge_paths = [] # [pair_num, max_edge_path_num, 2, max_edge_path_len]
        # edge_path_lengths = [] # [pair_num, max_edge_path_num, 2]
        # entPair2amrEdgePath = sample['entPair2amrEdgePath']
        # max_edge_path_num = max([len(item) for k,item in entPair2amrEdgePath.items()])
        
        # max_edge_path_len = 0
        # for k,item in entPair2amrEdgePath.items():
        #     for path_pair in item:
        #         for path in path_pair:
        #             if len(path) > max_edge_path_len:
        #                 max_edge_path_len = len(path)

        relNum = len(rel2id)
        multi_labels = []
        single_labels = []
        
        for pair in posPairs2label:
            pairs.append([pair[0],pair[1]])
            multi_label = [0] * relNum
            labels = posPairs2label[pair]
            for label in labels:
                multi_label[label] = 1
            multi_labels.append(multi_label)
            single_labels.append(random.choice(labels))

            # edge_path = entPair2amrEdgePath[(pair[0],pair[1])]
            # edge_path_padded = []
            # edge_path_length = []
            # for edge_path1,edge_path2 in edge_path:
            #     edge_path_length.append([len(edge_path1),len(edge_path2)])
            #     edge_path_padded.append([edge_path1 + [(0,0,0)] * (max_edge_path_len - len(edge_path1)), \
            #                                 edge_path2 + [(0,0,0)] * (max_edge_path_len - len(edge_path2))])
            # edge_path_padded += [[[[0,0,0]] * max_edge_path_len,[[0,0,0]] * max_edge_path_len] for i in range(max_edge_path_num - len(edge_path_padded))]
            # edge_path_length += [[0,0] for i in range(max_edge_path_num - len(edge_path_length))]
            # edge_paths.append(edge_path_padded)
            # edge_path_lengths.append(edge_path_length)
        
        for j,pair in enumerate(naPairs):
            multi_label = [0] * relNum
            multi_label[0] = 1
            multi_labels.append(multi_label)
            single_labels.append(0)
            pairs.append(pair)

            # edge_path = entPair2amrEdgePath[(pair[0],pair[1])]
            # edge_path_padded = []
            # edge_path_length = []
            # for edge_path1,edge_path2 in edge_path:
            #     edge_path_length.append([len(edge_path1),len(edge_path2)])
            #     edge_path_padded.append([edge_path1 + [[0,0,0]] * (max_edge_path_len - len(edge_path1)), \
            #                                 edge_path2 + [[0,0,0]] * (max_edge_path_len - len(edge_path2))])
            # edge_path_padded += [[[[0,0,0]] * max_edge_path_len,[[0,0,0]] * max_edge_path_len] for i in range(max_edge_path_num - len(edge_path_padded))]
            # edge_path_length += [[0,0] for i in range(max_edge_path_num - len(edge_path_length))]
            # edge_paths.append(edge_path_padded)
            # edge_path_lengths.append(edge_path_length)
        

        batched_label.append(single_labels)
        batched_multi_label.append(multi_labels)
        batched_entPair.append(pairs)
        batched_adj_matrix.append(sample['adjMatrix'])
        # batched_edge_path.append(edge_paths)
        # batched_edge_path_length.append(edge_path_lengths)
    
    max_pair_num = max([len(labels) for labels in batched_label])
    batched_label = [item + [-1] * (max_pair_num - len(item)) for item in batched_label]
    pad_multi_label = [0] * relNum
    batched_multi_label = [item + [pad_multi_label for i in range(max_pair_num - len(item))] for item in batched_multi_label]
    batched_entPair = [item + [[0,0] for i in range(max_pair_num - len(item))] for item in batched_entPair]
    
    batched_token_id = torch.LongTensor(batched_token_id)
    batched_node_span_pos = torch.LongTensor(batched_node_pos)
    batched_node_span_mask = torch.LongTensor(batched_node_span_mask)
    batched_node_mask = torch.BoolTensor(batched_node_mask)
    batched_node_ner = torch.LongTensor(batched_node_ner)
    batched_label = torch.LongTensor(batched_label)
    batched_multi_label = torch.LongTensor(batched_multi_label)
    batched_entPair = torch.LongTensor(batched_entPair)
    batched_label_mask = batched_label != -1
    batched_ent2MentionId = torch.LongTensor(batched_ent2MentionId)
    batched_ent2MentionId_mask = torch.BoolTensor(batched_ent2MentionId_mask)
    batched_adj_matrix = [torch.LongTensor(item) for item in batched_adj_matrix]
    # batched_edge_path = [torch.LongTensor(item) for item in batched_edge_path]
    # batched_edge_path_length = [torch.LongTensor(item) for item in batched_edge_path_length]

    return dict(title=batched_titles,       # list: [bsz]
                token_id=batched_token_id,  # Tensor: [bsz, max_token_len]
                graph=batched_graph,        # list(HeteroGraph): [bsz]
                node_span_pos=batched_node_span_pos, # Tensor: [bsz, max_node_num, max_span_len]
                node_span_mask=batched_node_span_mask, # Tensor: [bsz, max_node_num, max_span_len]
                node_mask=batched_node_mask, # Tensor: [bsz, max_node_num]
                node_ner=batched_node_ner, # Tensor: [bsz, max_node_num]
                label=batched_label,   # Tensor: [bsz, max_entPair]
                multi_label=batched_multi_label, # Tensor: [bsz, max_entPair, rel_num]
                label_mask=batched_label_mask, # Tensor: [bsz, max_entPair]
                ent_pair=batched_entPair, # Tensor: [bsz, max_entPair,2]
                # edge_path=batched_edge_path, # list(Tensor): [bsz] [pair_num, max_edge_path_num, 2, max_edge_path_len]
                # edge_path_length=batched_edge_path_length,  # list(Tensor): [bsz] [pair_num, max_edge_path_num, 2]
                ent2MentionId=batched_ent2MentionId, # Tensor: [bsz, max_ent_num, max_ent2mention_num]
                ent2MentionId_mask=batched_ent2MentionId_mask, # Tensor: [bsz, max_ent_num, max_ent2mention_num]
                adjMatrix=batched_adj_matrix
                )


def collate_fn_test(batch_samples):
    batched_token_id = []
    batched_mention_id = []
    batched_graph = []
    batched_labels = []
    batched_entPair = []
    batched_origPair = []
    batched_titles = []
    batched_label_mask = []
    batched_entNum = []

    max_span_len = max([max([len(item) for item in sample['graphData']['ndata']['span_pos']]) for sample in batch_samples])
    max_entPair_len = max([len(sample['entPairs']) for sample in batch_samples])
    for sample in batch_samples:
        batched_token_id.append(sample['tokenIds'])
        batched_mention_id.append(sample['mentionId'])
        batched_origPair.append(sample['origPairs'])
        batched_titles.append(sample['title'])
        batched_entNum.append(len(sample['vertexSet']))

        g = nx.DiGraph()
        g.add_nodes_from(sample['graphData']['nodes'])
        g.add_edges_from(sample['graphData']['edges'])
        dgl_g = dgl.from_networkx(g)
        dgl_g.ndata['ner_id'] = torch.LongTensor(sample['graphData']['ndata']['ner_id'])

        node_span_mask = []
        node_span_pos = []
        for span_pos_lst in sample['graphData']['ndata']['span_pos']:
            span_mask = [1] * len(span_pos_lst) + [0] * (max_span_len - len(span_pos_lst))
            span_pos_lst = span_pos_lst + [0] * (max_span_len - len(span_pos_lst))
            node_span_mask.append(span_mask)
            node_span_pos.append(span_pos_lst)
        dgl_g.ndata['span_pos'] = torch.LongTensor(node_span_pos)
        dgl_g.ndata['span_mask'] = torch.BoolTensor(node_span_mask)

        dgl_g.ndata['attr_id'] = torch.LongTensor(sample['graphData']['ndata']['attr_id'])
        dgl_g.edata['edge_id'] = torch.LongTensor(sample['graphData']['edata']['edge_id'])
        dgl_g.edata['norm'] = torch.LongTensor(sample['graphData']['edata']['norm'])
        # print(dgl_g.ndata['span_pos'].shape,dgl_g.ndata['span_mask'].shape,dgl_g.ndata['attr_id'].shape)
        assert(g.number_of_nodes()==dgl_g.num_nodes())

        batched_graph.append(dgl_g)

        batched_labels.append(sample['label_set'])
        label_mask = [1] * len(sample['entPairs']) + [0] * (max_entPair_len - len(sample['entPairs']))
        batched_label_mask.append(label_mask)
        entPairs = sample['entPairs'] + [[0,0] for i in range(max_entPair_len - len(sample['entPairs']))]
        batched_entPair.append(entPairs)
    
    batched_token_id = torch.LongTensor(batched_token_id)
    batched_mention_id = torch.LongTensor(batched_mention_id)
    batched_entPair = torch.LongTensor(batched_entPair)
    batched_label_mask = torch.LongTensor(batched_label_mask)

    return dict(title=batched_titles,token_id=batched_token_id,mention_id=batched_mention_id,graph=batched_graph,label=batched_labels,label_mask=batched_label_mask,ent_num=batched_entNum,ent_pair=batched_entPair,orig_pair=batched_origPair)



# Adapted from GAIN-BERT

class BertUtils():
    MASK = '[MASK]'
    CLS = "[CLS]"
    SEP = "[SEP]"

    def __init__(self, model_name, model_path=None):
        super().__init__()
        self.model_name = model_name
        print(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.max_len = 512

    def tokenize(self, text, masked_idxs=None):
        tokenized_text = self.tokenizer.tokenize(text)
        if masked_idxs is not None:
            for idx in masked_idxs:
                tokenized_text[idx] = self.MASK
        # prepend [CLS] and append [SEP]
        # see https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py#L195  # NOQA
        tokenized = [self.CLS] + tokenized_text + [self.SEP]
        return tokenized

    def tokenize_to_ids(self, text, masked_idxs=None, pad=True):
        tokens = self.tokenize(text, masked_idxs)
        return tokens, self.convert_tokens_to_ids(tokens, pad=pad)

    def convert_tokens_to_ids(self, tokens, pad=True):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor([token_ids])
        # assert ids.size(1) < self.max_len
        ids = ids[:, :self.max_len]  # https://github.com/DreamInvoker/GAIN/issues/4
        if pad:
            padded_ids = torch.zeros(1, self.max_len).to(ids)
            padded_ids[0, :ids.size(1)] = ids
            mask = torch.zeros(1, self.max_len).to(ids)
            mask[0, :ids.size(1)] = 1
            return padded_ids, mask
        else:
            return ids

    def flatten(self, list_of_lists):
        for list in list_of_lists:
            for item in list:
                yield item

    def subword_tokenize(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the special symbols required
                by Bert (CLS and SEP).
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
        """
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        subwords = [self.CLS] + list(self.flatten(subwords))[:509] + [self.SEP]
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_start_idxs[token_start_idxs > 509] = 512
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subword IDs, including IDs of the special
                symbols (CLS and SEP) required by Bert.
            - A mask indicating padding tokens.
            - An array of indices into the list of subwords. See
                doc of subword_tokenize.
        """
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        return subword_ids.numpy(), token_start_idxs, subwords

    def segment_ids(self, segment1_len, segment2_len):
        ids = [0] * segment1_len + [1] * segment2_len
        return torch.tensor([ids])
