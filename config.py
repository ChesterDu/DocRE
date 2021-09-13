
import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',default=20,type=int)
    parser.add_argument('--epoch',default=20,type=int)
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
    parser.add_argument('--use_loss_weight',action='store_true')

    parser.add_argument('--pred_activation',default='relu',type=str,choices=['relu','leaky relu','tanh','sigmoid','gelu'])
    parser.add_argument('--gnn_activation',default='relu',type=str,choices=['relu','leaky relu','tanh','sigmoid','gelu'])


    parser.add_argument('--lr',default=1e-3,type=float)
    parser.add_argument('--clip',default=1.0,type=float)
    parser.add_argument('--num_acumulation',default=1,type=int)
    parser.add_argument('--total_steps',default=10000,type=int)
    parser.add_argument('--metric_check_freq',default=100,type=int)
    parser.add_argument('--loss_print_freq',default=10,type=int)
    parser.add_argument('--device',default=0,type=int)

    parser.add_argument('--log_pth',default='../logs/',type=str)
    parser.add_argument('--checkpoint_pth',default='../checkpoints/',type=str)

    config = parser.parse_args(args=[])

    return config

def print_config(config):

    print('==========Training Configuration============')

    print("--seed {}".format(config.seed))
    print('--total_steps {}'.format(config.total_steps))
    print('--lr {}'.format(config.lr))
    print('--clip {}'.format(config.clip))
    print('--num_acumulation {}'.format(config.num_acumulation))
    print('--train_batch_size {}'.format(config.train_batch_size))
    print('--eval_batch_size {}\n'.format(config.eval_batch_size))

    print('--max_token_len {}'.format(config.max_token_len))
    print('--naPairs_num {}\n'.format(config.naPairs_num))

    print('--embed_type {}'.format(config.embed_type))
    print('--embed_pool_method {}'.format(config.embed_pool_method))
    print('--node_span_pool_method {}\n'.format(config.node_span_pool_method))

    print('--node_dim {}'.format(config.node_dim))
    print('--node_out_dim {}'.format(config.node_out_dim))
    print('--edge_dim {}'.format(config.edge_dim))
    print('--edge_type_emb_dim {}'.format(config.edge_type_emb_dim))
    print('--node_ner_emb_dim {}'.format(config.node_ner_emb_dim))
    print('--node_attr_emb_dim {}\n'.format(config.node_attr_emb_dim))

    print('--pred_activation {}'.format(config.pred_activation))
    print('--gnn_activation {}\n'.format(config.gnn_activation))

    print('--M {}'.format(config.M))
    print('--K {}'.format(config.K))
    print('--L {}\n'.format(config.L))

    print('--use_ner_feature {}'.format(config.use_ner_feature))
    print('--use_attr_feature {}'.format(config.use_attr_feature))
    print('--use_loss_weight {}'.format(config.use_loss_weight))

    print("=================================================")




