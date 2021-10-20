## Adapted from GAIN-BERT Implementation
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}






class multiLayerRelGraphConv(nn.Module):
    def __init__(self,L,node_in_dim,node_dim,node_out_dim,rel_names,num_bases,weight=True,bias=True,activation=None,self_loop=False,drop_out=0.0):
        super(multiLayerRelGraphConv, self).__init__()
        self.gnns = nn.ModuleList([RelGraphConvLayer(node_dim,node_dim,rel_names,num_bases=num_bases,weight=weight,bias=bias,\
                                                     activation=activation,self_loop=self_loop,dropout=drop_out) for i in range(L)])

        self.out_dim = node_out_dim
        self.node_in_fc = nn.Linear(node_in_dim,node_dim)
        for p in self.node_in_fc.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p,gain=nn.init.calculate_gain('relu'))
            else:
                nn.init.zeros_(p)
        self.node_out_fc = nn.Linear(node_dim,node_out_dim)

        for p in self.node_out_fc.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p,gain=nn.init.calculate_gain('relu'))
            else:
                nn.init.zeros_(p)
        if drop_out > 0.0:
            self.dropout = nn.Dropout(drop_out)
        else:
            self.dropout = None
        self.activation = activation

    def forward(self,g,node_in_features):
        node_features = node_in_features
        # node_features = self.node_in_fc(node_in_features)
        # if self.activation is not None:
        #     node_features = self.activation(node_features)
        # if self.dropout is not None:
        #     node_features = self.dropout(node_features)


        out_feature_list = [node_in_features.unsqueeze(0)]
        for gnn in self.gnns:
            node_features = gnn(g,{'nodes':node_features})['nodes']
            out_feature_list.append(node_features.unsqueeze(0))

        node_out_features = node_features

        # node_out_features = self.node_out_fc(node_features)
        # if self.activation is not None:
        #     node_out_features = self.activation(node_out_features)
        # if self.dropout is not None:
        #     node_out_features = self.dropout(node_out_features)

        return out_feature_list,node_out_features


