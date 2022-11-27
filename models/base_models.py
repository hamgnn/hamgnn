"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder,DotPredictor,MLPPredictor
# import layers.hyp_layers as hyp_layers
# import layers.hyp_layers_v1 as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1
import layers.ode_map
from layers.ode_map import *
import dgl
class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))

        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj,):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj,)
        return h
    def plotcur(self, x, adj):
        h = self.encoder.plotcur(x,adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        self.args =args
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        if self.manifold.name == 'Freemanifold':
            h = h
        else:


            h = self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.c), c=self.c)
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):

        idx = data[f'idx_{split}']
        # print('embeddings: ', embeddings)
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        # print('output: ', output)
        # print('labels: ', data['labels'][idx])
        # loss = F.nll_loss(output, data['labels'][idx], self.weights)
        # global ricci
        if self.args.odemap in ['ricci','riccilearn'] :
            # print(self.args.odemap)
            ricci_value = layers.ode_map.ricci
            ricci_value = torch.tanh(ricci_value)
            loss1 = F.nll_loss(output, data['labels'][idx], self.weights)
            loss = loss1 + 1*ricci_value
            print("ricci value: ", ricci_value)
            print("F.nll_loss value: ", loss1)
        else:
            loss = F.nll_loss(output, data['labels'][idx], self.weights)

        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.args=args
        # self.pred = DotPredictor()
        self.pred =MLPPredictor(args.dim)

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        # if self.manifold.name == 'Hyperboloid':
        #     h = self.manifold.proj_tan0(self.encoder.logmap0(h, c=self.c), c=self.c)
        # h = self.normalize(h)
        # emb_in = h[idx[:, 0], :]
        # emb_out = h[idx[:, 1], :]
        # # sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        # # probs = self.dc.forward(sqdist)
        # # print("emb_in shape: ", emb_in.shape)
        # # print("emb_out shape: ", emb_out.shape)
        # # probs = torch.mm(emb_in,emb_out,)
        # # print("probs shape: ", probs.shape)
        # # train_g = dgl.graph((idx[:, 0], idx[:, 1]), num_nodes= self.args.n_nodes)
        # # probs=self.pred(train_g, h)
        # sqdist_e = torch.sqrt((emb_in - emb_out).pow(2).sum(dim=-1) + 1e-15)
        # probs = self.dc.forward(sqdist_e)
        h = self.normalize(h)
        emb_in = h[idx[:, 0], :].clone()
        emb_out = h[idx[:, 1], :].clone()
        sqdist=torch.sqrt((emb_in - emb_out).pow(2).sum(dim=-1) + 1e-15)
        assert torch.max(sqdist) >= 0
        probs = self.dc.forward(sqdist)

        return probs

    def compute_metrics(self, embeddings, data, split):
        # if split == 'train':
        #     edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        # else:
        #     edges_false = data[f'{split}_edges_false']
        # pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        # neg_scores = self.decode(embeddings, edges_false)
        # assert not torch.isnan(pos_scores).any()
        # assert not torch.isnan(neg_scores).any()
        # # loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        # # loss += F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        # loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        # loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        # if pos_scores.is_cuda:
        #     pos_scores = pos_scores.cpu()
        #     neg_scores = neg_scores.cpu()
        # labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        # preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        # roc = roc_auc_score(labels, preds)
        # ap = average_precision_score(labels, preds)
        # metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        # return metrics
        torch.autograd.set_detect_anomaly(True)
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)

        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss_total = loss + F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        # loss = loss + 0.1*self.my_loss(embeddings,data,split,args)
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss_total, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

    def normalize(self, p):
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)
        return p

