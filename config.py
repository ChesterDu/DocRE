
import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',default=20,type=int)
    parser.add_argument('--train_batch_size',default=4,type=int)
    parser.add_argument('--eval_batch_size',default=8,type=int)
    parser.add_argument('--max_token_len',default=400,type=int)
    parser.add_argument('--naPairs_num',default=25,type=int)

    parser.add_argument('--embed_type',default='bert-base',type=str,choices=['bert-base','Elmo'])
    parser.add_argument('--embed_pth',default='../pretrained_embed/bert-base-uncased',type=str)
    parser.add_argument('--embed_pool_method',default='avg',type=str,choices=['avg','first','last'])

    parser.add_argument("--node_span_pool_method",default='avg',type=str,choices=['avg','first','last'])

    parser.add_argument('--node_dim',default=300,type=int)
    parser.add_argument('--node_out_dim',default=768,type=int)
    parser.add_argument('--edge_dim',default=300,type=int)
    parser.add_argument('--edge_type_emb_dim',default=768,type=int)
    parser.add_argument('--node_ner_emb_dim',default=300,type=int)
    parser.add_argument('--node_attr_emb_dim',default=300,type=int)

    parser.add_argument('--M',default=3,type=int)
    parser.add_argument('--K',default=3,type=int)
    parser.add_argument('--L',default=2,type=int)

    parser.add_argument('--use_ner_feature',action='store_true')
    parser.add_argument('--use_attr_feature',action='store_true')

    parser.add_argument('--pred_activation',default='relu',type=str,choices=['relu','leaky relu','tanh','sigmoid','relu'])
    parser.add_argument('--gnn_activation',default='relu',type=str,choices=['relu','leaky relu','tanh','sigmoid','relu'])


    parser.add_argument('--lr',default=1e-3,type=float)
    parser.add_argument('--clip',default=1.0,type=float)
    parser.add_argument('--num_acumulation',default=1,type=int)
    parser.add_argument('--total_steps',default=10000,type=int)
    parser.add_argument('--metric_check_freq',default=100,type=int)
    parser.add_argument('--loss_print_freq',default=10,type=int)
    parser.add_argument('--device',default=0,type=int)

    parser.add_argument('--log_pth',default='../logs/',type=str)
    parser.add_argument('--checkpoint_pth',default='../checkpoints/',type=str)

    config = parser.parse_args()

    return config