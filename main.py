import argparse
from random import shuffle

from trainner import Trainner
from model import finalModel,make_embed
from data import graphDataset, build_vocab, collate_fn
from torch.utils.data import Dataloader
import torch

## Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--seed',default=20,type=int)
parser.add_argument('--train_batch_size',default=4,type=int)
parser.add_argument('--num_acumulation',default=1,type=int)
parser.add_argument('--eval_batch_size',default=8,type=int)
parser.add_argument('--max_token_len',default=400,type=int)
parser.add_argument('--embed_type',default='bert-base',typ=str)
parser.add_argument('--embed_pth',default='pretrained_embed/bert-base-uncased')
parser.add_argument('--node_dim',default=300,type=int)
parser.add_argument('--node_out_dim',default=768,type=int)
parser.add_argument('--edge_in_dim',default=768,type=int)
parser.add_argument('--edge_dim',default=300,type=int)
parser.add_argument('--M',default=3,type=int)
parser.add_argument('--K',default=3,type=int)
parser.add_argument('--L',default=2,type=int)
parser.add_argument('--lr',default=1e-4,type=float)
parser.add_argument('--total_steps',default=10000,type=int)
parser.add_argument('--metric_check_freq',default=100,type=int)
parser.add_argument('--loss_print_freq',default=10,type=int)
parser.add_argument('--device',default=0,type=int)

parser.add_argument('--log_pth',default='logs/',type=str)
parser.add_argument('--checkpoint_pth',default='checkpoints/',type=str)

args = parser.parse_args()


## Make dataloader
train_data_pth = "DocRED/data/train_amr_annotated_full.json"
test_data_pth = "DocRED/data/test_amr_annotated_full.json"
dev_data_pth = "DocRED/data/dev_amr_annotated_full.json"
vocab = build_vocab(train_data_pth,test_data_pth,dev_data_pth)

train_dataset = graphDataset(train_data_pth,vocab,seed=args.seed,max_token_len=args.max_token_len,ignore_label_idx=-1,split='train')
# test_dataset = graphDataset(test_data_pth,vocab,seed=args.seed,max_token_len=args.max_token_len,ignore_label_idx=-1,split='test')
dev_dataset = graphDataset(dev_data_pth,vocab,seed=args.seed,max_token_len=args.max_token_len,ignore_label_idx=-1,split='dev')

train_loader = Dataloader(train_dataset, num_workers=2, batch_size=args.train_batch_size, shuffle=True, collate_fn = collate_fn)
# test_loader = Dataloader(test_dataset, num_workers=2, batch_size=args.eval_batch_size, shuffle=True, collate_fn= collate_fn)
dev_loader = Dataloader(dev_dataset, num_workers=2, batch_size=args.train_batch_size, shuffle=True, collate_fn= collate_fn)

## Make model
embed_layer,node_in_dim = make_embed(vocab,args.embed_type,args.embed_pth)
node_dim = args.node_dim
node_out_dim = args.node_out_dim
edge_in_dim = args.edge_in_dim
edge_dim = args.edge_dim
M = args.M
K = args.K
L = args.L
device = args.device

model = finalModel(embed_layer,node_in_dim,node_dim,node_out_dim,edge_in_dim,edge_dim,M,K,L).to(device)

## Make Optimizer and Criterion
optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

## Make Trainner
trainner = Trainner(args,model,optimizer,criterion)
trainner.train(train_loader,dev_loader)