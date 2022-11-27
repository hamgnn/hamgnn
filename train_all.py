from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import dgl.graph_index
import numpy as np
import optimizers
import manifolds
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics

import scipy.sparse as sp
import dgl
import statistics
def torch_sparse_to_coo(adj):
    m_index = adj._indices().cpu().numpy()
    row = m_index[0]
    col = m_index[1]
    data = adj._values().cpu().numpy()
    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(adj.size()[0], adj.size()[1]))
    return sp_matrix

def feat_preprocess(features, feat_norm=None, device='cpu'):
    r"""

    Description
    -----------
    Preprocess the features.

    Parameters
    ----------
    features : torch.Tensor or numpy.array
        Features in form of torch tensor or numpy array.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    Returns
    -------
    features : torch.Tensor
        Features in form of torch tensor on chosen device.

    """

    def feat_normalize(feat, norm=None):
        if norm == "arctan":
            feat = 2 * np.arctan(feat) / np.pi
        elif norm == "tanh":
            feat = torch.tanh(feat)
        else:
            feat = feat

        return feat

    if type(features) != torch.Tensor:
        features = torch.FloatTensor(features)
    elif features.type() != 'torch.FloatTensor':
        features = features.float()
    if feat_norm is not None:
        features = feat_normalize(features, norm=feat_norm)

    features = features.to(device)

    return features


def train(args):
    test_acc_clean = []
    test_score_attack = []
    # randseed=[1234,2589,3698,1478,4569]
    for runtime in range(4):

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if int(args.double_precision):
            torch.set_default_dtype(torch.float64)
        if int(args.cuda) >= 0:
            torch.cuda.manual_seed(args.seed)
        args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
        args.patience = args.epochs if not args.patience else  int(args.patience)
        logging.getLogger().setLevel(logging.INFO)
        if args.save:
            if not args.save_dir:
                dt = datetime.datetime.now()
                date = f"{dt.year}_{dt.month}_{dt.day}"
                models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
                save_dir = get_dir_name(models_dir)
            else:
                save_dir = args.save_dir
            logging.basicConfig(level=logging.INFO,
                                handlers=[
                                    logging.FileHandler(os.path.join(save_dir, 'log_'+args.dataset+args.odemap+str(args.cuda)+str(args.task)+'.txt')),
                                    logging.StreamHandler()
                                ])

        logging.info(f'Using: {args.device}')
        logging.info("Using seed {}.".format(args.seed))

        # Load data
        # data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
        data = load_data(args, os.path.join('./data', args.dataset))

        # a=feat_preprocess(data['features'],feat_norm='arctan')
        # data['features']=feat_preprocess(data['features'],feat_norm='arctan')
        args.n_nodes, args.feat_dim = data['features'].shape
        if args.task == 'nc':
            Model = NCModel
            args.n_classes = int(data['labels'].max() + 1)
            logging.info(f'Num classes: {args.n_classes}')
            print("""----Data statistics inintial dataset------'
                                      #Nodes %d
                                      #numfeatures %d
                                      #Train samples %d
                                      #Val samples %d
                                      #Test samples %d""" %
                  (data['features'].shape[0], data['features'].shape[1],
                   len(data['idx_train']),
                   len(data['idx_val']),
                   len(data['idx_test'])))
        else:
            args.nb_false_edges = len(data['train_edges_false'])
            args.nb_edges = len(data['train_edges'])
            if args.task == 'lp':
                Model = LPModel
            else:
                Model = RECModel
                # No validation for reconstruction task
                args.eval_freq = args.epochs + 1

        if not args.lr_reduce_freq:
            args.lr_reduce_freq = args.epochs

        # Model and optimizer
        model = Model(args)
        logging.info(str(model))

        odelist = []
        for name, para in model.named_parameters():
            if 'odefunc' in name:
                odelist.append(name)

        ode_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in odelist, model.named_parameters()))))
        base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in odelist, model.named_parameters()))))

        optimizer = getattr(optimizers, args.optimizer)( [{'params':base_params,'lr':args.lr}, {'params':ode_params,'lr': args.lr}],
                                                        weight_decay=args.weight_decay)

        # adj = torch_sparse_to_coo(data['adj_train_norm'])
        # g = dgl.from_scipy(adj, idtype=torch.int32, device=args.cuda)
        # print(" got g")
        # transform = dgl.RandomWalkPE(k=16)
        # g = transform(g)
        # h = g.ndata['PE']
        #
        # # h = dgl.laplacian_pe(g, 2).to(self.device)
        # print("position encoding dim: ", h.shape)
        h = None
        # data['features'] = data['features'] + h


        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.lr_reduce_freq),
            gamma=float(args.gamma)
        )
        tot_params = sum([np.prod(p.size()) for p in model.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")
        if args.cuda is not None and int(args.cuda) >= 0 :
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
            model = model.to(args.device)
            for x, val in data.items():
                if torch.is_tensor(data[x]):
                    data[x] = data[x].to(args.device)
        # Train model
        t_total = time.time()
        counter = 0
        best_val_metrics = model.init_metric_dict()
        best_test_metrics = None
        best_emb = None
        train_loss_list=[]
        train_acc_list =[]
        train_f1_list =[]
        val_loss_list = []
        val_acc_list = []
        val_f1_list = []
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            train_metrics = model.compute_metrics(embeddings, data, 'train')
            train_metrics['loss'].backward()
            # torch.save(model.encoder, os.path.join(save_dir,
            #                                        'model_' + str(args.dataset) + args.odemap + str(args.cuda) + str(
            #                                            args.task) + 'model.pth'))

            train_loss = train_metrics['loss'].item()
            if args.task =='nc':
                train_acc = train_metrics['acc']
                train_f1 = train_metrics['f1']
            else:
                train_acc = train_metrics['roc']
                train_f1 = train_metrics['ap']


            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            optimizer.step()
            lr_scheduler.step()
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                       'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                       format_metrics(train_metrics, 'train'),
                                       'time: {:.4f}s'.format(time.time() - t)
                                       ]))
            if (epoch + 1) % args.eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    embeddings = model.encode(data['features'], data['adj_train_norm'])
                    val_metrics = model.compute_metrics(embeddings, data, 'val')
                    val_loss = val_metrics['loss'].item()
                    if args.task == 'nc':
                        val_acc = val_metrics['acc']
                        val_f1 = val_metrics['f1']
                    else:
                        val_acc = val_metrics['roc']
                        val_f1 = val_metrics['ap']


                    train_loss_list.append(train_loss)
                    train_acc_list.append(train_acc)
                    train_f1_list.append(train_f1)
                    val_loss_list.append(val_loss)
                    val_acc_list.append(val_acc)
                    val_f1_list.append(val_f1)

                if (epoch + 1) % args.log_freq == 0:
                    logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))

                if model.has_improved(best_val_metrics, val_metrics):
                    best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                    logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(best_test_metrics, 'test')]))
                    best_emb = embeddings.cpu()
                    if args.save:
                        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                    best_val_metrics = val_metrics
                    counter = 0
                else:
                    counter += 1
                    # print('counter of patience: ', counter)
                    if counter == args.patience and epoch > args.min_epochs:
                        logging.info("Early stopping")

                        break

        logging.info("Optimization Finished!")
        logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        if not best_test_metrics:
            model.eval()
            with torch.no_grad():
                best_emb = model.encode(data['features'], data['adj_train_norm'])
                best_test_metrics = model.compute_metrics(best_emb, data, 'test')
        test_acc_clean.append(best_test_metrics['acc'])
        test_score_attack.append(best_test_metrics['f1'])
        logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
        logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
        if args.save:
            np.save(os.path.join(save_dir, 'embeddings_'+str(args.dataset)+args.odemap+str(args.cuda)+str(args.task)+'.npy'), best_emb.cpu().detach().numpy())
            if hasattr(model.encoder, 'att_adj'):
                filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
                pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
                print('Dumped attention adj: ' + filename)

            json.dump(vars(args), open(os.path.join(save_dir, 'config_'+str(args.dataset)+args.odemap+str(args.cuda)+str(args.task)+'.json'), 'w'))
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_'+str(args.dataset)+args.odemap+str(args.cuda)+str(args.task)+'.pth'))
            torch.save(model.encoder, os.path.join(save_dir, 'model_' + str(args.dataset) + args.odemap + str(args.cuda) + str(args.task) + 'model.pth'))
            logging.info(f"Saved model in {save_dir}")

        # with open('traindata_'+args.dataset+args.odemap+str(args.cuda)+str(args.task)+'.pkl', 'wb') as f:
        #     pickle.dump([train_loss_list,train_acc_list,train_f1_list],f)
        #
        # f.close()
        # with open('valdata_'+args.dataset+args.odemap+str(args.cuda)+str(args.task)+'.pkl', 'wb') as f1:
        #     pickle.dump([val_loss_list,val_acc_list,val_f1_list],f1)
        #
        # f1.close()
        args.seed = args.seed - 1
        print("test_acc_clean: ", test_acc_clean)
    logging.info("*" * 80)
    logging.info("test_acc_clean: ", )
    logging.info(test_acc_clean)
    logging.info("test_f1", )
    logging.info(test_score_attack)
    logging.info("Mean of test_acc_clean: ", )
    logging.info(statistics.mean(test_acc_clean))
    logging.info("Std of test_acc_clean: ")
    logging.info(statistics.stdev(test_acc_clean))
    logging.info("Mean of test_f1: ", )
    logging.info(statistics.mean(test_score_attack))
    logging.info("Std of test_f1: ", )
    logging.info(statistics.stdev(test_score_attack))
    print("test_acc_clean: ", test_acc_clean)
    print("test_acc_clean mean: ", statistics.mean(test_acc_clean))
    print("test_acc_clean var: ", statistics.stdev(test_acc_clean))



if __name__ == '__main__':
    args = parser.parse_args()
    CUDA_LAUNCH_BLOCKING = 1
    train(args)
