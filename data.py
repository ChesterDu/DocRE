from fastNLP.core.vocabulary import Vocabulary
import networkx as nx
import json
import torch
import fastNLP
import dgl

with open("/content/drive/MyDrive/DocRE/DocRED/rel2id.json","r") as fp:
    rel2id = json.load(fp)
with open("/content/drive/MyDrive/DocRE/DocRED/ner2id.json","r") as fp:
    ner2id = json.load(fp)

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
    def __init__(self,data_pth,vocab,max_token_len=256,max_sen_num=10,ignore_label_idx=-1,split='train'):
        super(graphDataset,self).__init__()

        with open(data_pth,'r') as fp:
            self.samples = json.load(fp)
        
        self.create_amr_graph_alighments()
        
        self.vocab = vocab
        self.max_token_len = 256
        self.max_sen_num = 10
        self.split = split
        self.get_token_ids()

        self.max_label_num = max([len(sample['labels']) for sample in self.samples])
        self.ignore_label_idx = ignore_label_idx
        self.create_labels()

    def __getitem__(self,index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)
    
    def get_token_ids(self):
    # sents is a list of list of tokens
        for i,sample in enumerate(self.samples):
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

            self.samples[i]['tokenIds'] = token_ids
            self.samples[i]['senStartPos'] = sen_start_pos_lst
    
    def create_labels(self):
        for i,sample in enumerate(self.samples):
            labels = []
            headEntNodes = []
            tailEntNodes = []
            entid2node = sample['Entid2Node']
            for i, label_data in enumerate(sample['labels']):
                labels.append(rel2id[label_data['r']])
                headEntNodes.append(entid2node[label_data['h']])
                tailEntNodes.append(entid2node[label_data['t']])

            labels += [self.ignore_label_idx] * (self.max_label_num - len(labels))
            headEntNodes += [0] * (self.max_label_num - len(headEntNodes))
            tailEntNodes += [0] * (self.max_label_num - len(tailEntNodes))

            self.samples[i]['headEntNodes'] = headEntNodes
            self.samples[i]['tailEntNodes'] = tailEntNodes
            self.samples[i]['finalLabels'] = labels

    
    ## Given the AMR Parsing results, create amr graph alignments to the tokens
    def create_amr_graph_alighments(self):
        ## Function that find amr node span id
        def find_amr_span_id(amr_node_id,amr_graph):
        #     print(amr_graph['alignments'])
            amr_node_id = str(amr_node_id)
            try:
                amr_span_end_id = amr_graph['alignments'][amr_node_id][1] - 1 
                amr_span_start_id = amr_graph['alignments'][amr_node_id][0] - 1
            except:
                amr_span_end_id = amr_graph['alignments'][amr_node_id] - 1
                amr_span_start_id = amr_graph['alignments'][amr_node_id] - 1
            
            return amr_span_start_id,amr_span_end_id+1

        for i,sample in enumerate(self.samples):
        # sample = samples[0]
            G = nx.DiGraph()
            amr_graphs = sample['amrGraphs']
            unique_id = 1
            amrId_to_uniqueId = {}
            for sent_id,amr_graph in enumerate(amr_graphs):
                for u,rel,v in amr_graph['edges']:
                    u_amrId = str(sent_id)+'-'+str(u)
                    v_amrId = str(sent_id)+'-'+str(v)
                    if u_amrId not in amrId_to_uniqueId:
                        try:
                            temp = amr_graph['alignments'][str(u)]
                        except:
                            continue
                        amr_node_pos = find_amr_span_id(u,amr_graph)
                        amr_node_span = ' '.join(sample['sents'][sent_id][amr_node_pos[0]:amr_node_pos[1]])
                        G.add_node(unique_id,attr='amr',ent_type='NONE',span=amr_node_span,sent_id=sent_id,pos=amr_node_pos)
                        amrId_to_uniqueId[str(sent_id)+'-'+str(u)] = unique_id
                        unique_id += 1
                    if v_amrId not in amrId_to_uniqueId:
                        try:
                            temp = amr_graph['alignments'][str(v)]
                        except:
                            continue
                        amr_node_pos = find_amr_span_id(v,amr_graph)
                        amr_node_span = ' '.join(sample['sents'][sent_id][amr_node_pos[0]:amr_node_pos[1]])
                        G.add_node(unique_id,attr='amr',ent_type='NONE',span=amr_node_span,sent_id=sent_id,pos=amr_node_pos)
                        amrId_to_uniqueId[str(sent_id)+'-'+str(v)] = unique_id
                        unique_id += 1
                    
                    if rel[1:].startswith("ARG") and rel[1:].endswith("-of"):
                        G.add_edge(amrId_to_uniqueId[v_amrId],amrId_to_uniqueId[u_amrId],edge_type='AMR-'+rel[0:5],edge_id=get_edge_idx(rel[1:5]))
                    
                    else:
                        G.add_edge(amrId_to_uniqueId[u_amrId],amrId_to_uniqueId[v_amrId],edge_type='AMR-'+rel,edge_id=get_edge_idx(rel[1:]))

            entid2node = {}
            for i,ent in enumerate(sample['vertexSet']):
                G.add_node(unique_id,attr='entity',ent_type=ent[0]['type'],span=ent[0]['name'],sent_id=[item['sent_id'] for item in ent],pos=[item['pos'] for item in ent])
                entid2node[i] = unique_id
                ent_id = unique_id
                unique_id += 1
                for mention in ent:
                    mention_span = ' '.join(sample['sents'][mention['sent_id']][mention['pos'][0]:mention['pos'][1]])
                    G.add_node(unique_id,attr='mention',ent_type=mention['type'],span=mention_span,sent_id=mention['sent_id'],pos=mention['pos'])
                    mention_idx = mention['pos'][1] - 1
                    G.add_edge(ent_id,unique_id,edge_type='ENT-MENTION',edge_id=get_edge_idx('ENT-MENTION'))
        #             G.add_edge(unique_id,ent_id,rel='MENTION-ENT')

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
                        
                        G.add_edge(unique_id,nearest_graph_id,edge_type='MENTION-NEAREST',edge_id=get_edge_idx('MENTION-NEAREST'))

                    unique_id += 1
            self.samples[i]['graphData'] = dict(nodes=[[n, G.nodes[n]] for n in G.nodes()],edges=[[u, v, G.edges[(u,v)]] for u,v in G.edges()])
            self.samples[i]['Entid2Node'] = entid2node



def collate_fn(batch_samples):
    batched_token_id = []
    batched_sen_start_pos = []
    batched_graph = []
    batched_label = []
    batched_headEnt = []
    batched_tailEnt = []

    for sample in batch_samples:
        batched_token_id.append(sample['tokenIds'])
        batched_sen_start_pos.append(sample['senStartPos'])

        g = dgl.DGLGraph()
        g.add_nodes(len(sample['graphData']['nodes']))
        g.add_edges([u for u,_,_ in sample['graphData']['edges']],[v for _,v,_ in sample['graphData']['edges']])
        batched_graph.append(g)

        batched_label.append(sample['finalLabels'])
        batched_headEnt.append(sample['headEntNodes'])
        batched_tailEnt.append(sample['tailEntNodes'])
    
    batched_token_id = torch.LongTensor(batched_token_id)
    batched_sen_start_pos = torch.LongTensor(batched_sen_start_pos)
    batched_graph = dgl.batch(batched_graph)
    batched_label = torch.LongTensor(batched_label)
    batched_headEnt = torch.LongTensor(batched_headEnt)
    batched_tailEnt = torch.LongTensor(batched_tailEnt)

    return dict(token_id=batched_token_id,sen_start_pos=batched_sen_start_pos,graph=batched_graph,label=batched_label,headEnt=batched_headEnt,tailEnt=batched_tailEnt)



        