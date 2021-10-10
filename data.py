from os import initgroups
from fastNLP.core.vocabulary import Vocabulary
import networkx as nx
import json
from networkx.algorithms.assortativity import pairs
import torch
import pickle
import dgl
import random
import numpy as np
import os
from collections import defaultdict
import tqdm

with open("../DocRED/rel2id.json","r") as fp:
    rel2id = json.load(fp)
with open("../DocRED/ner2id.json","r") as fp:
    ner2id = json.load(fp)

attr2id = {'entity':0,"mention":1,"amr":2}
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
        elif edge_type_str == "ENT-MENTION":
            return 11
        elif edge_type_str == "MENTION-AMR":
            return 12
        elif edge_type_str == "MENTION-NEAREST":
            return 13
        else:
            return 14


def build_vocab(train_data_pth,test_data_pth,dev_data_pth):
    vocab = Vocabulary(min_freq=50)
    def add_words_to_vocab(data_pth,vocab):
        with open(data_pth,'r') as fp:
            samples = json.load(fp)
        for sample in samples:
            for sen in sample['sents']:
                vocab.add_word_lst(sen)
        return vocab
    
    vocab = add_words_to_vocab(train_data_pth,vocab)
    vocab = add_words_to_vocab(test_data_pth,vocab)
    vocab = add_words_to_vocab(dev_data_pth,vocab)

    return vocab

class graphDataset(torch.utils.data.Dataset):
    def __init__(self,config,processed_data_pth,raw_data_pth,vocab,ignore_label_idx=-1,split='train',fact_in_train=set([])):
        super(graphDataset,self).__init__()

        random.seed(config.seed)

        with open(raw_data_pth,'r') as fp:
            self.samples = json.load(fp)
            if config.debug:
                self.samples = self.samples[:20]
        
        # self.create_amr_graph_alighments()
        # print("AMR Graph Alignments Completed!")
        
        self.vocab = vocab
        self.max_token_len = config.max_token_len
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
        print("Split: {} || Sample Num: {} || Max Token Num: {} || Max Ent Num: {} || Max Pair Num: {}".format(self.split,len(self.samples),self.max_token_len,self.max_ent_num,self.max_pair_num))

    def manipulate_naPairs_num(self):
        if self.split == 'train':
            for doc_id,doc in enumerate(self.samples):
                h_t_pair2label = self.samples[doc_id]['posPairs2label']
                self.samples[doc_id]['naPairs_num'] = int(max([10,self.naPairs_alpha * len(h_t_pair2label)]))
        else:
            for doc_id,doc in enumerate(self.samples):
                self.samples[doc_id]['naPairs_num'] = 0
            


    def process_data(self):
        # bar = tqdm.tqdm(total=len(self.samples))
        # bar.update(0)
        for doc_id,doc in enumerate(self.samples):
            ## convert word to ids
            word_id = []
            sen_start_pos_lst = [0]
            for i, sen in enumerate(doc['sents']):
                sen_id = [self.vocab.to_index(word) for word in sen]
                sen_start_pos = sen_start_pos_lst[-1] + len(sen_id)
                sen_start_pos_lst.append(sen_start_pos)
                word_id += sen_id
            
            word_id += [0] * (self.max_token_len - len(word_id))
            sen_start_pos_lst = sen_start_pos_lst[:-1]
            
            self.samples[doc_id]['tokenIds'] = word_id

            ## create entity-mention graphs
            G = nx.DiGraph()
            span_node_index = np.zeros(self.max_token_len)
            node_ner_id = []
            node_span_pos = []
            node_span_mask = []
            node_attr_id = []
            # edge_type_id = []
            node_id = 0
            entity2node = {}
            entity2mentino_nodes=[]

            ## TODO: attach AMR graphs
            if self.use_amr_graph:
                amr_graphs = doc['amrGraphs']
                amr_id2node_id = {}
                for sent_id,amr_graph in enumerate(amr_graphs):
                    alignments = amr_graph['alignments']
                    for u,rel,v in amr_graph['edges']:
                        u_amrId = str(sent_id) + "-" + str(u)
                        v_amrId = str(sent_id) + "-" + str(v)
                        if u_amrId not in amr_id2node_id:
                            G.add_node(node_id)
                            node_attr_id.append(attr2id['amr'])
                            node_ner_id.append(len(ner2id))  ## additional id for amr node
                            temp = np.zeros(self.max_token_len)
                            try:
                                amr_span_id_lst = np.array(alignments[str(u)]) - 1 + sen_start_pos_lst[sent_id]    ## amr parsing results index start from 1
                                temp[amr_span_id_lst] = 1
                                node_span_mask.append(list(temp))
                                if amr_span_id_lst.ndim == 0:
                                    node_span_pos.append([int(amr_span_id_lst)])
                                else:
                                    node_span_pos.append([int(item) for item in list(amr_span_id_lst)])
                            except KeyError:
                                node_span_mask.append([0] * self.max_token_len)
                                node_span_pos.append([])
                            

                            amr_id2node_id[u_amrId] = node_id
                            node_id += 1
                        if v_amrId not in amr_id2node_id:
                            G.add_node(node_id)
                            node_attr_id.append(attr2id['amr'])
                            node_ner_id.append(len(ner2id))  ## additional id for amr node
                            temp = np.zeros(self.max_token_len)
                            try:
                                amr_span_id_lst = np.array(alignments[str(v)]) - 1 + sen_start_pos_lst[sent_id]    ## amr parsing results index start from 1
                                temp[amr_span_id_lst] = 1
                                node_span_mask.append(list(temp))
                                if amr_span_id_lst.ndim == 0:
                                    node_span_pos.append([int(amr_span_id_lst)])
                                else:
                                    node_span_pos.append([int(item) for item in list(amr_span_id_lst)])
                            except KeyError:
                                node_span_mask.append([0] * self.max_token_len)
                                node_span_pos.append([])


                            amr_id2node_id[v_amrId] = node_id
                            node_id += 1

                        if rel[1:].startswith("ARG") and rel[1:].endswith("-of"):
                            G.add_edge(amr_id2node_id[v_amrId],amr_id2node_id[u_amrId],edge_type=get_edge_idx(rel[1:5]))
                            # edge_type_id.append(get_edge_idx(rel[1:5]))
                        
                        else:
                            G.add_edge(amr_id2node_id[u_amrId],amr_id2node_id[v_amrId],edge_type=get_edge_idx(rel[1:]))
                            # edge_type_id.append(get_edge_idx(rel[1:]))

            for ent_id, mentions in enumerate(doc['vertexSet']):
                entity2node[ent_id] = node_id
                entity2mentino_nodes.append([])
                ent_node_id = node_id
                G.add_node(node_id)
                node_attr_id.append(attr2id['entity'])
                node_span_pos.append([])
                node_span_mask.append([0] * self.max_token_len)
                node_ner_id.append(ner2id[mentions[0]['type']])
                node_id += 1
                for men_id, mention in enumerate(mentions):
                    entity2mentino_nodes[ent_id].append(node_id)
                    men_start_pos = mention['pos'][0] + sen_start_pos_lst[mention['sent_id']]
                    men_end_pos = mention['pos'][1] + sen_start_pos_lst[mention['sent_id']]
                    mention['globalPos'] = [men_start_pos,men_end_pos]
                    span_node_index[men_start_pos:men_end_pos] = node_id
                    G.add_node(node_id)
                    node_attr_id.append(attr2id['mention'])
                    node_span_pos.append(list(range(men_start_pos,men_end_pos)))
                    temp = np.zeros(self.max_token_len)
                    temp[men_start_pos:men_end_pos] = 1
                    node_span_mask.append(list(temp))
                    node_ner_id.append(ner2id[mention['type']])
                    # edge_type_id.append(get_edge_idx('ENT-MENTION'))
                    G.add_edge(ent_node_id,node_id,edge_type=get_edge_idx('ENT-MENTION'))
                    
                    # Find AMR node alignment
                    if self.use_amr_graph:
                        amr_graph = amr_graphs[mention['sent_id']]

                        mention_pos = men_end_pos - 1
                        match_flag = False
                        for amr_node_id in amr_graph['nodes']:
                            amr_id = str(mention['sent_id']) + "-" + amr_node_id
                            try:
                                amr_node_id = amr_id2node_id[amr_id]
                            except:
                                continue
                            span_mask = node_span_mask[amr_node_id]
                            amr_span_index = np.nonzero(np.array(span_mask) == 1)
                            if np.sum(amr_span_index) == 0:
                                continue
                            amr_span_start_pos,amr_span_end_pos = int(np.min(amr_span_index)),int(np.max(amr_span_index))
                            if (mention_pos >= amr_span_start_pos) and (mention_pos <= amr_span_end_pos):
                                G.add_edge(node_id,amr_node_id,edge_type=get_edge_idx('MENTION-AMR'))
                                # edge_type_id.append(get_edge_idx('MENTION-AMR'))
                                match_flag = True
                                break
                        
                        if match_flag == False:       ## Find nearest amr node
                            min_dist = 100000000
                            min_dist_amr_node = -1
                            for amr_node_id in amr_graph['nodes']:
                                amr_id = str(mention['sent_id']) + "-" + amr_node_id
                                try:
                                    amr_node_id = amr_id2node_id[amr_id]
                                except:
                                    continue
                                span_mask = node_span_mask[amr_node_id]
                                amr_span_index = np.nonzero(np.array(span_mask) == 1)
                                if np.sum(amr_span_index) == 0:
                                    continue
                                amr_span_start_pos,amr_span_end_pos = int(np.min(amr_span_index)),int(np.max(amr_span_index))

                                dist = abs(mention_pos - amr_span_start_pos) + abs(mention_pos - amr_span_end_pos)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_dist_amr_node = amr_node_id
                            
                            if min_dist_amr_node != -1:
                                G.add_edge(node_id,min_dist_amr_node,edge_type=get_edge_idx('MENTION-NEAREST'))
                                # edge_type_id.append(get_edge_idx('MENTION-NEAREST'))

                    node_id += 1
                            
            ## create edge norm for rgcn
            for u in G.nodes():
                count_dict = defaultdict(int)
                for v in G.adj[u]:
                    count_dict[G.edges[u,v]['edge_type']] += 1
                for v in G.adj[u]:
                    G.edges[u,v]['norm'] = 1 / count_dict[G.edges[u,v]['edge_type']]
            
            edge_norm = [G.edges[u,v]['norm'] for u,v in G.edges()]
            edge_type = [G.edges[u,v]['edge_type'] for u,v in G.edges()]
            node_data = dict(ner_id=node_ner_id,span_pos=node_span_pos,attr_id=node_attr_id)
            edge_data = dict(edge_id=edge_type,norm=edge_norm)
            self.samples[doc_id]['graphData'] = dict(nodes=[n for n in G.nodes()],edges=[[u, v] for u,v in G.edges()],ndata=node_data,edata=edge_data)
            self.samples[doc_id]['mentionId'] = list(span_node_index)

            ## create label, head and tail entities
            node2entity = {entity2node[k]:k for k in entity2node.keys()}

            for i,label_data in enumerate(doc['labels']):
                label_data['in_train'] = False
                for n1 in doc['vertexSet'][label_data['h']]:
                    for n2 in doc['vertexSet'][label_data['t']]:
                        if self.split == 'train':
                            self.fact_in_train.add((n1['name'],n2['name'],label_data['r']))
                        else:
                            if (n1['name'],n2['name'],label_data['r']) in self.fact_in_train:
                                label_data['in_train'] = True
                                break
            if self.split in ['train','dev']:
                h_t_pair2label = {}
                for i,label_data in enumerate(doc['labels']):
                    pair = (entity2node[label_data['h']],entity2node[label_data['t']])
                    if pair not in h_t_pair2label:
                        h_t_pair2label[pair] = [rel2id[label_data['r']]]
                    else:
                        h_t_pair2label[pair].append(rel2id[label_data['r']])
                    
                entNum = len(doc['vertexSet'])
                naPairs = []
                for i in range(entNum):
                    for j in range(entNum):
                        if i==j:
                            continue
                        pair = (entity2node[i],entity2node[j])
                        if pair not in h_t_pair2label:
                            naPairs.append(pair)
                            # labels.append(0)
                random.shuffle(naPairs)
                orig_h_t_pairs = [[node2entity[item[0]],node2entity[item[1]]] for item in list(h_t_pair2label.keys()) + naPairs]
                self.samples[doc_id]['posPairs2label'] = h_t_pair2label
                self.samples[doc_id]['naPairs'] = naPairs
                self.samples[doc_id]['naPairs_num'] = int(max([10,self.naPairs_alpha * len(h_t_pair2label)]))
                if self.split == 'dev':
                    self.samples[doc_id]['naPairs_num'] = 0
            else:
                label_set = {}
                for label in doc['labels']:
                    head, tail, relation, intrain = label['h'], label['t'], label['r'], label['in_train']
                    label_set[(head, tail, rel2id[relation])] = intrain

                orig_h_t_pairs = []
                entPairs = []
                entNum = len(doc['vertexSet'])
                for h_idx in range(entNum):
                    for t_idx in range(entNum):
                        if h_idx == t_idx:
                            continue
                        orig_h_t_pairs.append([h_idx,t_idx])
                        entPairs.append([entity2node[h_idx],entity2node[t_idx]])
                self.samples[doc_id]['entPairs'] = entPairs
                self.samples[doc_id]['label_set'] = label_set

            
            self.samples[doc_id]['origPairs'] = orig_h_t_pairs
            # bar.update(1)



def collate_fn(batch_samples):  
    batched_token_id = []
    batched_mention_id = []
    batched_graph = []
    batched_label = []
    batched_multi_label = []
    batched_entPair = []
    batched_origPair = []
    batched_titles = []

    max_span_len = max([max([len(item) for item in sample['graphData']['ndata']['span_pos']]) for sample in batch_samples])
    for sample in batch_samples:
        batched_token_id.append(sample['tokenIds'])
        batched_mention_id.append(sample['mentionId'])
        batched_origPair.append(sample['origPairs'])
        batched_titles.append(sample['title'])

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
        
        naPairs_num = min([len(sample['naPairs']), sample['naPairs_num']])
        if sample['naPairs_num'] == 0:
            naPairs_num = len(sample['naPairs'])
        
        naPairs = sample['naPairs'][:naPairs_num]
        posPairs = list(sample['posPairs2label'].keys())
        pairs = posPairs + naPairs

        relNum = len(rel2id)
        multi_labels = []
        single_labels = []
        for i,pair in enumerate(posPairs):
            multi_label = [0] * relNum
            labels = sample['posPairs2label'][pair]
            for label in labels:
                multi_label[label] = 1
            multi_labels.append(multi_label)
            single_labels.append(random.choice(labels))
        
        for j,pair in enumerate(naPairs):
            multi_label = [0] * relNum
            multi_label[0] = 1
            multi_labels.append(multi_label)
            single_labels.append(0)

        batched_label.append(single_labels)
        batched_multi_label.append(multi_labels)
        batched_entPair.append(pairs)
    
    max_pair_num = max([len(labels) for labels in batched_label])
    batched_label = [item + [-1] * (max_pair_num - len(item)) for item in batched_label]
    pad_multi_label = [0] * relNum
    batched_multi_label = [item + [pad_multi_label for i in range(max_pair_num - len(item))] for item in batched_multi_label]
    batched_entPair = [item + [[0,0] for i in range(max_pair_num - len(item))] for item in batched_entPair]
    
    batched_token_id = torch.LongTensor(batched_token_id)
    batched_mention_id = torch.LongTensor(batched_mention_id)
    batched_label = torch.LongTensor(batched_label)
    batched_multi_label = torch.LongTensor(batched_multi_label)
    batched_entPair = torch.LongTensor(batched_entPair)
    batched_label_mask = batched_label != -1

    return dict(title=batched_titles,token_id=batched_token_id,mention_id=batched_mention_id,graph=batched_graph,label=batched_label,multi_label=batched_multi_label,label_mask=batched_label_mask,ent_pair=batched_entPair,orig_pair=batched_origPair)


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



