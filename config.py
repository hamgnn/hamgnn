import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.01, 'learning rate'),
        'dropout': (0.2, 'dropout probability'),
        'cuda': (2, 'which cuda device to use (-1 for cpu training)'),

        'epochs': (100, 'maximum number of epochs to train for'),

        'weight-decay': (0.001, 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'momentum': (0.999, 'momentum in optimizer'),

        'patience': (200, 'patience for early stopping'),

        'seed': (1234, 'seed for training'),
        'log-freq': (10, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': ('./experiment3', 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (300, 'do not early stop before min-epochs')
    },
    'model_config': {
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc]'),

        'model': ('GeoGCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN,GeoGCN]'),

        'dim': (64, 'embedding dimension'),
        'manifold': ('Freemanifold', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall,Freemanifold]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (3, 'number of hidden layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not'),

        'odemap': ('h1extend', 'which ode function to use, can be any of [linear,v1, v2, v1learn,v2learn,ricci,riccilearn,v1xlearn,v5,v5learn,v5extend]'),
        
        'acth2':('rehu', '[act1, act2, act3, rehu]'),


        'feat_h2_scale':(8, '[8,16,4], 8 default'),


        'logmethods': ('ode', 'which log to use, can be any of [ode, vanilla,]'),
        'kdim': (8, 'v embedding dimension'),
    },
    'data_config': {

        'dataset': ('cora', 'which dataset to use[coauthor, amazoncomputer,amazonphoto]'),

        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
