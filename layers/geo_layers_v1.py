"""Hyperbolic layers."""
import math

import dgl.graph_index
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt,SpGraphAttentionLayer,GraphAttentionLayer
import torch.nn as nn
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint
import geotorch

from pytorch_block_sparse import BlockSparseLinear
import sparselinear as sl
# from layers.ode_map import Connection_g,ODEfunc, Connectionnew,Connection,Connection_gnew,Connection_ricci,Connection_riccilearn,Connectionxlearn,Connection_v5,Connection_v5new
from layers.ode_map import *
from layers.H_1 import Hamilton,Hamilton_learn
from layers.H_2 import Hamilton_V2
from layers.H_3 import Hamilton_V3
from layers.H_4 import Hamilton_V4
from layers.H_5 import Hamilton_V5
from layers.H_6 import Hamilton_V6
from layers.H_7 import Hamilton_V7
from layers.H_8 import Hamilton_V8

from torch_geometric.nn import GATConv
from torch_geometric.utils import from_scipy_sparse_matrix
class ODEBlock(nn.Module):

    def __init__(self, odefunc,time):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor(time).float()

    def forward(self, x,):
        self.integration_time = self.integration_time.type_as(x)
        # out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3,method='rk4',options={'step_size':0.1})
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3, method='euler',
                     options={'step_size': 0.5})
        # out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3, method='implicit_adams',
        #              options={'step_size': 0.2})
        # out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3, method='dopri5',)
        # out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3, method='dopri8',)

        return out[1]




def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims = dims + [args.dim]
        acts = acts+ [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures







class GeoGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_features, out_features,  dropout, act, use_bias, use_att, local_agg,device,n_nodes,args):
        super(GeoGraphConvolution, self).__init__()



        self.agg = GeoAgg( out_features, dropout, use_att, local_agg,n_nodes, args.odemap,args)
        self.hyp_act = GeoAct( act,in_features,)
        self.device = device

    def forward(self, input):
        x, adj, = input
        h = self.agg.forward(x, adj)
        h = self.hyp_act.forward(h,adj)
        output = h, adj
        return output

class GeoAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self,  in_features, dropout, use_att, local_agg,n_nodes,odemethod,args):
        super(GeoAgg, self).__init__()


        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        self.args = args
        if self.use_att:
            # self.att = DenseAtt(in_features, dropout)
            # self.att =SpGraphAttentionLayer(in_features,in_features,dropout)
            self.att = GraphAttentionLayer(in_features, in_features, dropout=args.dropout, activation=F.elu, alpha=args.alpha, nheads=args.n_heads, concat=0)
            # self.att = GATConv(in_features, in_features, heads=args.n_heads,concat=False, dropout=0.6)
            # self.att.to(args.cuda)
        self.weight = nn.Parameter(torch.Tensor(in_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(in_features))
        self.weight_k = nn.Parameter(torch.Tensor(in_features*args.kdim, in_features))

        self.reset_parameters()

        # self.v = nn.Parameter(torch.Tensor(n_nodes, in_features), requires_grad=True)
        self.v = nn.Parameter(torch.randn(n_nodes, in_features), requires_grad=True)
        self.v_k = nn.Parameter(torch.randn(n_nodes, in_features*args.kdim), requires_grad=True)
        # print("self.v shape:",self.v.shape)
        # self.reset_parameters()

        self.odemethod = odemethod
        if self.odemethod == 'v1':
            self.odefunc = Connection(int(in_features / 2))

        elif self.odemethod == 'v5learn':
            self.odefunc = Connection_v5new(in_features, self.v)
        elif self.odemethod == 'linear':
            self.odefunc = ODEfunc(in_features)
        elif self.odemethod == 'h1extend':
            self.odefunc = Hamilton(int(in_features))
        elif self.odemethod == 'h1learn':
            self.odefunc = Hamilton_learn(int(in_features), self.v)
        elif self.odemethod == 'h2extend':
            self.odefunc = Hamilton_V2(int(in_features))
        elif self.odemethod == 'h2learn':
            self.odefunc = Hamilton_V2(int(in_features))
        elif self.odemethod == 'h3extend':
            self.odefunc = Hamilton_V3(int(in_features),args.kdim)
        elif self.odemethod == 'h3new':
            self.odefunc = Hamilton_V3(int(in_features),args.kdim)
        elif self.odemethod == 'h4extend':
            self.odefunc = Hamilton_V4(int(in_features))
        elif self.odemethod == 'h4learn':
            self.odefunc = Hamilton_V4(int(in_features))
        elif self.odemethod == 'h5extend':
            self.odefunc = Hamilton_V5(int(in_features))
        elif self.odemethod == 'h5learn':
            self.odefunc = Hamilton_V5(int(in_features))
        elif self.odemethod == 'h6extend':
            self.odefunc = Hamilton_V6(int(in_features))
        elif self.odemethod == 'h6learn':
            self.odefunc = Hamilton_V6(int(in_features))
        elif self.odemethod == 'h7extend':
            self.odefunc = Hamilton_V7(int(in_features))
        elif self.odemethod == 'h7learn':
            self.odefunc = Hamilton_V7(int(in_features))
        elif self.odemethod == 'h8extend':
            self.odefunc = Hamilton_V8(int(in_features))
        elif self.odemethod == 'h8learn':
            self.odefunc = Hamilton_V8(int(in_features))



        else:
            print(" -----no ode func------- ")
        self.odeblock_exp = ODEBlock(self.odefunc, [0, 1])


    # def reset_parameters(self):
    #     init.xavier_uniform_(self.v, gain=1)

    def forward(self, x, adj):
        #
        xt = x
        # if self.odemethod == 'v5extend' or 'h2extend':
        #     # vt =xt
        #     # vt = self.linear(xt)
        #     drop_weight =self.weight
        #
        #     vt = xt @ drop_weight.transpose(-1, -2)
        #     xt = torch.hstack([xt,vt])
        #     out = self.odeblock_exp(xt, )
        #     out = out[..., 0:int(self.in_features)]
        #
        #
        #
        # else:
        #     # print("x shape: ",xt.shape)
        #     out = self.odeblock_exp(xt, )

        # if self.odemethod == 'h1learn':
        if 'learn' in self.odemethod :
            xt = torch.hstack([xt, self.v])
            out = self.odeblock_exp(xt, )
            out = out[..., 0:int(self.in_features)]
        elif self.odemethod == 'h3extend':
            drop_weight = self.weight_k

            vt = xt @ drop_weight.transpose(-1, -2)
            xt = torch.hstack([xt, vt])
            out = self.odeblock_exp(xt, )
            out = out[..., 0:int(self.in_features)]

        elif self.odemethod == 'h3new':

            xt = torch.hstack([xt, self.v_k])
            out = self.odeblock_exp(xt, )
            out = out[..., 0:int(self.in_features)]

        else:
            drop_weight = self.weight

            vt = xt @ drop_weight.transpose(-1, -2)
            xt = torch.hstack([xt, vt])
            out = self.odeblock_exp(xt, )
            out = out[..., 0:int(self.in_features)]


        x_tangent = out
        if self.use_att:
            input_att = x_tangent, adj
            support_t ,_ = self.att(input_att)
            # support_t = self.att(x_tangent, from_scipy_sparse_matrix(adj)[0].to(self.args.cuda))

        else:
            support_t = torch.spmm(adj, x_tangent)
            # support_t = x_tangent

        output = support_t
        return output


    def reset_parameters(self):
        # init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.eye_(self.weight)
        init.constant_(self.bias, 0)
        init.eye_(self.weight_k)


class GeoAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self,  act,in_features,):
        super(GeoAct, self).__init__()

        self.act = act
        self.in_features =in_features
    def forward(self, x,adj,):
        # xt = self.act(x,)
        # xt = torch.sin(x)
        xt = x
        out = xt
        return out










