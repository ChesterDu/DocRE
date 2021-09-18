import argparse
import random
from random import shuffle
from collections import defaultdict
from config import parse_config, print_config
from trainner import Trainner
from model import finalModel
from data import graphDataset, build_vocab, collate_fn, rel2id
from opt import OpenAIAdam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch



config = parse_config()
print_config(config)


random.seed(config.seed)
torch.manual_seed(config.seed)
cudnn.deterministic = True

## Make dataloader
train_data_pth = "../DocRED/data/train_amr_annotated_full.json"
train_processed_data_pth = "../DocRED/data/train_amr_annotated_full_processed.json"
test_data_pth = "../DocRED/data/test_amr_annotated_full.json"
test_processed_data_pth = "../DocRED/data/test_amr_annotated_full_processed.json"
dev_data_pth = "../DocRED/data/dev_amr_annotated_full.json"
dev_processed_data_pth = "../DocRED/data/dev_amr_annotated_full_processed.json"
vocab = build_vocab(train_data_pth,test_data_pth,dev_data_pth)

train_dataset = graphDataset(config,train_processed_data_pth,train_data_pth,vocab,ignore_label_idx=-1,split='train')
# test_dataset = graphDataset(test_data_pth,vocab,seed=config.seed,max_token_len=config.max_token_len,ignore_label_idx=-1,split='test')
dev_dataset = graphDataset(config,dev_processed_data_pth,dev_data_pth,vocab,ignore_label_idx=-1,split='dev')

train_loader = DataLoader(train_dataset, num_workers=2, batch_size=config.train_batch_size, shuffle=True, collate_fn = collate_fn)
# test_loader = DataLoader(test_dataset, num_workers=2, batch_size=config.eval_batch_size, shuffle=True, collate_fn= collate_fn)
dev_loader = DataLoader(dev_dataset, num_workers=2, batch_size=config.train_batch_size, shuffle=True, collate_fn= collate_fn)

## Make model
model = finalModel(vocab,config)
# model = debugModel(embed_layer,node_in_dim,node_dim,node_out_dim,edge_in_dim,edge_dim,M,K,L).to(device)

## Make Optimizer and Criterion
# optimizer = torch.optim.Adam(model.parameters(),lr=config.lr)
optimizer = OpenAIAdam(model.parameters(),
                                  lr=config.lr,
                                  schedule='warmup_linear',
                                  warmup=0.002,
                                  t_total=config.total_steps,
                                  b1=0.9,
                                  b2=0.999,
                                  e=1e-08,
                                  l2=0.01,
                                  vector_l2=True,
                                  max_grad_norm=config.clip)
# optimizer = torch.optim.SGD(model.parameters(),lr=config.lr)

if config.use_loss_weight:
    label_count = defaultdict(int)
    total_label = 0
    for sample in train_dataset.samples:
        entPair = []
        for label_data in sample['labels']:
            pair = [label_data['h'],label_data['t']]
            if pair not in entPair:
                entPair.append(pair)
                label_count[rel2id[label_data['r']]] += 1
        
        num_ent = len(sample['vertexSet'])
        num_na = num_ent * (num_ent - 1) - len(entPair)

        label_count[0] += num_na
        total_label += num_ent * (num_ent - 1)

    label_weight_lst = []
    for k in label_count:
        label_weight_lst.append(1 / label_count[k])

    label_weight = torch.FloatTensor(label_weight_lst).to(config.device)
    # criterion = torch.nn.CrossEntropyLoss(weight=label_weight,ignore_index=-1)
    criterion = torch.nn.BCEWithLogitsLoss(weight=label_weight,reduction='none')
else:
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

## Make Trainner
trainner = Trainner(config,model,optimizer,criterion)
trainner.train(train_loader,dev_loader)