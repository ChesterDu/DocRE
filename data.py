from fastNLP.core.vocabulary import Vocabulary
import networkx as nx
import json
from networkx.algorithms.assortativity import pairs
import torch
import fastNLP
import dgl
import random
import numpy as np

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
    def __init__(self,data_pth,vocab,seed=25,max_token_len=256,ignore_label_idx=-1,split='train'):
        super(graphDataset,self).__init__()

        with open(data_pth,'r') as fp:
            self.samples = json.load(fp)
        
        # self.create_amr_graph_alighments()
        # print("AMR Graph Alignments Completed!")
        
        self.vocab = vocab
        self.max_token_len = max_token_len
        self.max_sen_num = max([len(sample['sents']) for sample in self.samples])
        self.split = split
        # self.get_token_ids()

        self.max_ent_num = max([len(sample['vertexSet']) for sample in self.samples])
        self.max_pair_num = self.max_ent_num * (self.max_ent_num - 1)
        self.ignore_label_idx = ignore_label_idx
        # self.create_labels()

        self.process_data()
        self.print_stat()
        

        random.seed(seed)
        random.shuffle(self.samples)

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
            node_attr_id = []
            edge_type_id = []
            node_id = 0
            entity2node = []
            entity2mentino_nodes=[]

            for ent_id, mentions in enumerate(doc['vertexSet']):
                entity2node.append(node_id)
                entity2mentino_nodes.append([])
                ent_node_id = node_id
                G.add_node(node_id)
                node_attr_id.append(attr2id['entity'])
                node_span_pos.append([-1,-1])
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
                    node_span_pos.append([men_start_pos,men_end_pos])
                    node_ner_id.append(ner2id[mention['type']])
                    G.add_edge(ent_node_id,node_id)
                    edge_type_id.append(get_edge_idx('ENT-MENTION'))
                    node_id += 1

            ## TODO: attach AMR graphs
            node_data = dict(ner_id=node_ner_id,span_pos=node_span_pos,attr_id=node_attr_id)
            edge_data = dict(edge_id=edge_type_id)
            self.samples[doc_id]['graphData'] = dict(nodes=[n for n in G.nodes()],edges=[[u, v] for u,v in G.edges()],ndata=node_data,edata=edge_data)
            self.samples[doc_id]['mentionId'] = list(span_node_index)

            ## create label, head and tail entities
            labels = []
            entPairs = []
            for i,label_data in enumerate(doc['labels']):
                pair = [entity2node[label_data['h']],entity2node[label_data['t']]]
                labels.append(rel2id[label_data['r']])
                entPairs.append(pair)
            
            entNum = len(doc['vertexSet'])
            for i in range(entNum):
                for j in range(entNum):
                    if i==j:
                        continue
                    pair = [entity2node[i],entity2node[j]]
                    if pair not in entPairs:
                        entPairs.append(pair)
                        labels.append(0)
            
            labels += [-1] * (self.max_pair_num - len(labels))
            entPairs += [[0,0] for i in range(self.max_pair_num-len(entPairs))]

            self.samples[doc_id]['finalLabels'] = labels
            self.samples[doc_id]['entPairs'] = entPairs



    def get_token_ids(self):
    # sents is a list of list of tokens
        for j,sample in enumerate(self.samples):
            token_ids = []
            sen_start_pos_lst = [0]   # starting ids of each sentence 
            for i,sen in enumerate(sample['sents']):
                temp = [self.vocab.to_index(word) for word in sen]
                start_pos = sen_start_pos_lst[-1] + len(temp)
                sen_start_pos_lst.append(start_pos)
                token_ids += temp

            token_ids += [0] * (self.max_token_len - len(token_ids))
            sen_start_pos_lst = sen_start_pos_lst[:-1]
            sen_start_pos_lst += [-1] * (self.max_sen_num - len(sen_start_pos_lst))

            self.samples[j]['tokenIds'] = token_ids
            self.samples[j]['senStartPos'] = sen_start_pos_lst
    
    def create_labels(self):
        for idx,sample in enumerate(self.samples):
            labels = []
            headEntNodes = []
            tailEntNodes = []
            entid2node = sample['Entid2Node']
            posPairSet = []
            for i, label_data in enumerate(sample['labels']):
                labels.append(rel2id[label_data['r']])
                headEntNodes.append(entid2node[label_data['h']])
                tailEntNodes.append(entid2node[label_data['t']])
                posPairSet.append((label_data['h'],label_data['t']))
            
            entNum = len(sample['vertexSet'])
            for i in range(entNum):
                for j in range(entNum):
                    if i==j:
                        continue
                    if (i,j) not in posPairSet:
                        headEntNodes.append(entid2node[i])
                        tailEntNodes.append(entid2node[j])
                        labels.append(0)

            labels += [self.ignore_label_idx] * (self.max_label_num - len(labels))
            headEntNodes += [0] * (self.max_label_num - len(headEntNodes))
            tailEntNodes += [0] * (self.max_label_num - len(tailEntNodes))

            self.samples[idx]['headEntNodes'] = headEntNodes
            self.samples[idx]['tailEntNodes'] = tailEntNodes
            self.samples[idx]['finalLabels'] = labels

    
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
        dgl_g.ndata['attr_id'] = torch.LongTensor(sample['graphData']['ndata']['attr_id'])
        dgl_g.edata['edge_id'] = torch.LongTensor(sample['graphData']['edata']['edge_id'])
        assert(g.number_of_nodes()==dgl_g.num_nodes())

        batched_graph.append(dgl_g)

        batched_label.append(sample['finalLabels'])
        batched_entPair.append(sample['entPairs'])
    
    batched_token_id = torch.LongTensor(batched_token_id)
    batched_mention_id = torch.LongTensor(batched_mention_id)
    batched_label = torch.LongTensor(batched_label)
    batched_entPair = torch.LongTensor(batched_entPair)

    return dict(token_id=batched_token_id,mention_id=batched_mention_id,graph=batched_graph,label=batched_label,entPair=batched_entPair)



        