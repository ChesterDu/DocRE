from os import initgroups
from fastNLP.core.vocabulary import Vocabulary
import networkx as nx
import json
from networkx.algorithms.assortativity import pairs
import torch
import fastNLP
import dgl
import random
import numpy as np
import os
from collections import defaultdict

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
    vocab = Vocabulary()
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
    def __init__(self,config,processed_data_pth,raw_data_pth,vocab,ignore_label_idx=-1,split='train'):
        super(graphDataset,self).__init__()

        random.seed(config.seed)

        with open(raw_data_pth,'r') as fp:
            self.samples = json.load(fp)
        
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

        self.naPairs_num = config.naPairs_num
        # self.create_labels()

        if os.path.exists(processed_data_pth):
            with open(processed_data_pth,'r') as fp:
                self.samples = json.load(fp)
        else:
            self.process_data()
            with open(processed_data_pth,'w') as fp:
                json.dump(self.samples,fp)
        self.print_stat()


    def __getitem__(self,index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)
    
    def print_stat(self):
        print("==============Data Statistic===================")
        print("Split: {} || Sample Num: {} || Max Token Num: {} || Max Ent Num: {} || Max Pair Num: {}".format(self.split,len(self.samples),self.max_token_len,self.max_ent_num,self.max_pair_num))

    def process_data(self):
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
            edge_type_id = []
            node_id = 0
            entity2node = []
            entity2mentino_nodes=[]

            ## TODO: attach AMR graphs
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
                        except:
                            node_span_mask.append([0] * self.max_token_len)
                        

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
                        except:
                            node_span_mask.append([0] * self.max_token_len)


                        amr_id2node_id[v_amrId] = node_id
                        node_id += 1

                    if rel[1:].startswith("ARG") and rel[1:].endswith("-of"):
                        G.add_edge(amr_id2node_id[v_amrId],amr_id2node_id[u_amrId])
                        edge_type_id.append(get_edge_idx(rel[1:5]))
                    
                    else:
                        G.add_edge(amr_id2node_id[u_amrId],amr_id2node_id[v_amrId])
                        edge_type_id.append(get_edge_idx(rel[1:]))

            for ent_id, mentions in enumerate(doc['vertexSet']):
                entity2node.append(node_id)
                entity2mentino_nodes.append([])
                ent_node_id = node_id
                G.add_node(node_id)
                node_attr_id.append(attr2id['entity'])
                # node_span_pos.append([-1,-1])
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
                    # node_span_pos.append([men_start_pos,men_end_pos])
                    temp = np.zeros(self.max_token_len)
                    temp[men_start_pos:men_end_pos] = 1
                    node_span_mask.append(list(temp))
                    node_ner_id.append(ner2id[mention['type']])
                    edge_type_id.append(get_edge_idx('ENT-MENTION'))
                    
                    # Find AMR node alignment
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
                            G.add_edge(node_id,amr_node_id)
                            edge_type_id.append(get_edge_idx('MENTION-AMR'))
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
                            G.add_edge(node_id,min_dist_amr_node)
                            edge_type_id.append(get_edge_idx('MENTION-NEAREST'))

                    node_id += 1
                            

            ## TODO: attach AMR graphs
            for u in G.nodes():
                count_dict = defaultdict(int)
                for v in G.adj[u]:
                    count_dict[G.edges[u,v]['edge_type']] += 1
                for v in G.adj[u]:
                    G.edges[u,v]['norm'] = 1 / count_dict[G.edges[u,v]['edge_type']]
            
            edge_norm = [G.edges[u,v]['norm'] for u,v in G.edges()]
            node_data = dict(ner_id=node_ner_id,span_pos=node_span_pos,span_mask=node_span_mask,attr_id=node_attr_id)
            edge_data = dict(edge_id=edge_type_id,norm=edge_norm)
            self.samples[doc_id]['graphData'] = dict(nodes=[n for n in G.nodes()],edges=[[u, v] for u,v in G.edges()],ndata=node_data,edata=edge_data)
            self.samples[doc_id]['mentionId'] = list(span_node_index)

            ## create label, head and tail entities
            labels = []
            entPairs = []
            for i,label_data in enumerate(doc['labels']):
                pair = [entity2node[label_data['h']],entity2node[label_data['t']]]
                if pair not in entPairs:
                    labels.append(rel2id[label_data['r']])
                    entPairs.append(pair)
            
            entNum = len(doc['vertexSet'])
            naPairs = []
            for i in range(entNum):
                for j in range(entNum):
                    if i==j:
                        continue
                    pair = [entity2node[i],entity2node[j]]
                    if pair not in entPairs:
                        naPairs.append(pair)
                        # labels.append(0)
            
            # labels += [-1] * (self.max_pair_num - len(labels))
            # entPairs += naPairs
            # entPairs += [[0,0] for i in range(self.max_pair_num-len(entPairs))]

            self.samples[doc_id]['entLabels'] = labels
            self.samples[doc_id]['entPairs'] = entPairs
            self.samples[doc_id]['naPairs'] = naPairs
            self.samples[doc_id]['naPairs_num'] = self.naPairs_num
            if self.split in ['test','dev']:
                self.samples[doc_id]['naPairs_num'] = len(naPairs)


    
    ## Given the AMR Parsing results, create amr graph alignments to the tokens
    def create_amr_graph_alighments(self):
        ## Function that find amr node span id
        def find_amr_span_id(amr_node_id,amr_graph,sen_len):
        #     print(amr_graph['alignments'])
            amr_node_id = str(amr_node_id)
            try:
                amr_span_end_id = amr_graph['alignments'][amr_node_id][1] - 1 
                amr_span_start_id = amr_graph['alignments'][amr_node_id][0] - 1
            except:
                amr_span_end_id = amr_graph['alignments'][amr_node_id] - 1
                amr_span_start_id = amr_graph['alignments'][amr_node_id] - 1

            if amr_span_end_id == -2:
              amr_span_end_id = sen_len - 1
            if amr_span_start_id == -2:
              amr_span_start_id = sen_len - 1
            
            return amr_span_start_id,amr_span_end_id+1

        for j,sample in enumerate(self.samples):
            G = nx.DiGraph()
            unique_id = 1
            amr_graphs = []
            # amr_graphs = sample['amrGraphs']
            # amrId_to_uniqueId = {}
            # for sent_id,amr_graph in enumerate(amr_graphs):
            #     for u,rel,v in amr_graph['edges']:
            #         u_amrId = str(sent_id)+'-'+str(u)
            #         v_amrId = str(sent_id)+'-'+str(v)
            #         if u_amrId not in amrId_to_uniqueId:
            #             try:
            #                 temp = amr_graph['alignments'][str(u)]
            #             except:
            #                 continue
            #             amr_node_pos = find_amr_span_id(u,amr_graph,len(sample['sents'][sent_id]))
            #             amr_node_span = ' '.join(sample['sents'][sent_id][amr_node_pos[0]:amr_node_pos[1]])
            #             G.add_node(unique_id,attr='amr',ent_type='NONE',span=amr_node_span,sent_id=sent_id,pos=amr_node_pos)
            #             amrId_to_uniqueId[str(sent_id)+'-'+str(u)] = unique_id
            #             unique_id += 1
            #         if v_amrId not in amrId_to_uniqueId:
            #             try:
            #                 temp = amr_graph['alignments'][str(v)]
            #             except:
            #                 continue
            #             amr_node_pos = find_amr_span_id(v,amr_graph,len(sample['sents'][sent_id]))
            #             amr_node_span = ' '.join(sample['sents'][sent_id][amr_node_pos[0]:amr_node_pos[1]])
            #             G.add_node(unique_id,attr='amr',ent_type='NONE',span=amr_node_span,sent_id=sent_id,pos=amr_node_pos)
            #             amrId_to_uniqueId[str(sent_id)+'-'+str(v)] = unique_id
            #             unique_id += 1
                    
            #         if rel[1:].startswith("ARG") and rel[1:].endswith("-of"):
            #             G.add_edge(amrId_to_uniqueId[v_amrId],amrId_to_uniqueId[u_amrId],edge_type='AMR-'+rel[0:5],edge_id=get_edge_idx(rel[1:5]))
                    
            #         else:
            #             G.add_edge(amrId_to_uniqueId[u_amrId],amrId_to_uniqueId[v_amrId],edge_type='AMR-'+rel,edge_id=get_edge_idx(rel[1:]))

            entid2node = {}
            for i,ent in enumerate(sample['vertexSet']):
                G.add_node(unique_id,attr='entity',ent_type=ent[0]['type'],span=ent[0]['name'],sent_id=[item['sent_id'] for item in ent],pos=[item['pos'] for item in ent])
                # G.add_node(unique_id,attr='entity',ent_type=ent[0]['type'],span=ent[0]['name'],sent_id=-1,pos=[-1,-1])
                entid2node[i] = unique_id
                ent_id = unique_id
                unique_id += 1
                for mention in ent:
                    mention_span = ' '.join(sample['sents'][mention['sent_id']][mention['pos'][0]:mention['pos'][1]])
                    G.add_node(unique_id,attr='mention',ent_type=mention['type'],span=mention_span,sent_id=mention['sent_id'],pos=mention['pos'])
                    mention_idx = mention['pos'][1] - 1
                    G.add_edge(ent_id,unique_id,edge_type='ENT-MENTION',edge_id=get_edge_idx('ENT-MENTION'))
                    
                    if amr_graphs == []:
                      unique_id += 1
                      continue
                    amr_graph = amr_graphs[mention['sent_id']]
                    
                    match_flag = False
                    for amr_node_id in amr_graph['nodes']:
                        try:
                            amr_node_graph_id = amrId_to_uniqueId[str(mention['sent_id'])+'-'+amr_node_id]
                        except:
                            continue
                        amr_span_start_id,amr_span_end_id = G.nodes[amr_node_graph_id]['pos']

                        if mention_idx <= (amr_span_end_id-1) and mention_idx >= amr_span_start_id:
                            G.add_edge(unique_id,amr_node_graph_id,edge_type='MENTION-AMR',edge_id=get_edge_idx('MENTION-AMR'))
                            match_flag = True
                            break
                    
                    if not match_flag:
                        min_d = 100000
                        nearest_graph_id = -1
                        for amr_node_id in amr_graph['nodes']:
                            try:
                                amr_node_graph_id = amrId_to_uniqueId[str(mention['sent_id'])+'-'+amr_node_id]
                            except:
                                continue
                            amr_span_start_id,amr_span_end_id = G.nodes[amr_node_graph_id]['pos']
                            
                            d = abs(mention_idx - amr_span_start_id) + abs(mention_idx - amr_span_end_id)
                            if d < min_d:
                                nearest_graph_id = amr_node_graph_id
                                min_d = d
                        if nearest_graph_id != -1:
                            G.add_edge(unique_id,nearest_graph_id,edge_type='MENTION-NEAREST',edge_id=get_edge_idx('MENTION-NEAREST'))

                    unique_id += 1
            self.samples[j]['graphData'] = dict(nodes=[[n, G.nodes[n]] for n in G.nodes()],edges=[[u, v, G.edges[(u,v)]] for u,v in G.edges()])
            self.samples[j]['Entid2Node'] = entid2node



def collate_fn(batch_samples):  
    batched_token_id = []
    batched_mention_id = []
    batched_graph = []
    batched_label = []
    batched_entPair = []

    for sample in batch_samples:
        batched_token_id.append(sample['tokenIds'])
        batched_mention_id.append(sample['mentionId'])

        g = nx.DiGraph()
        g.add_nodes_from(sample['graphData']['nodes'])
        g.add_edges_from(sample['graphData']['edges'])
        dgl_g = dgl.from_networkx(g)
        dgl_g.ndata['ner_id'] = torch.LongTensor(sample['graphData']['ndata']['ner_id'])
        dgl_g.ndata['span_pos'] = torch.LongTensor(sample['graphData']['ndata']['span_pos'])
        dgl_g.ndata['span_mask'] = torch.BoolTensor(sample['graphData']['ndata']['span_mask'])
        dgl_g.ndata['attr_id'] = torch.LongTensor(sample['graphData']['ndata']['attr_id'])
        dgl_g.edata['edge_id'] = torch.LongTensor(sample['graphData']['edata']['edge_id'])
        dgl_g.edata['norm'] = torch.LongTensor(sample['graphData']['edata']['norm'])
        assert(g.number_of_nodes()==dgl_g.num_nodes())

        batched_graph.append(dgl_g)
        
        random.shuffle(sample['naPairs'])
        naPairs = sample['naPairs'][:sample['naPairs_num']]
        pairs = sample['entPairs'] + naPairs
        labels = sample['entLabels'] + [0] * len(naPairs)

        batched_label.append(labels)
        batched_entPair.append(pairs)
    
    max_pair_num = max([len(labels) for labels in batched_label])
    batched_label = [item + [-1] * (max_pair_num - len(item)) for item in batched_label]
    batched_entPair = [item + [[0,0] for i in range(max_pair_num - len(item))] for item in batched_entPair]
    
    batched_token_id = torch.LongTensor(batched_token_id)
    batched_mention_id = torch.LongTensor(batched_mention_id)
    batched_label = torch.LongTensor(batched_label)
    batched_entPair = torch.LongTensor(batched_entPair)

    return dict(token_id=batched_token_id,mention_id=batched_mention_id,graph=batched_graph,label=batched_label,ent_pair=batched_entPair)

