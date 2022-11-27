import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import AddSelfLoop
import argparse
import statistics

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(in_size, hid_size, heads[0], feat_drop=0.6, attn_drop=0.6, activation=F.elu))
        self.gat_layers.append(
            dglnn.GATConv(hid_size * heads[0], out_size, heads[1], feat_drop=0.6, attn_drop=0.6, activation=None))

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def my_loss(g, x, device):  # g is the graph and x is the output logits
    prob = F.softmax(x)
    num_nodes = g.number_of_nodes()

    degs = g.in_degrees().float()
    deg_mat = torch.diag(degs)  # get degree matrix to calculate laplacian

    # deg_mat = torch.eye(num_nodes,device=device)
    adj = g.adjacency_matrix(scipy_fmt="csr")
    laplacian =deg_mat - torch.from_numpy(adj.toarray()).to(device)

    y = torch.matmul(torch.matmul(torch.transpose(prob, 0, 1), laplacian), prob)
    y = torch.trace(y)
    return y / num_nodes


def my_loss_mask(g, x, train_mask,device):  # g is the graph and x is the output logits
    prob = F.softmax(x)
    num_nodes = g.number_of_nodes()

    degs = g.in_degrees().float()
    deg_mat = torch.diag(degs)  # get degree matrix to calculate laplacian

    # deg_mat = torch.eye(num_nodes,device=device)
    adj = g.adjacency_matrix(scipy_fmt="csr")
    # laplacian =deg_mat - torch.from_numpy(adj.toarray()).to(device)
    #
    # y = torch.matmul(torch.matmul(torch.transpose(prob, 0, 1), laplacian), prob)
    # y = torch.trace(y)

    loss_all = []
    for i in range(num_nodes):
        neigh_node = g.predecessors(i).long()
        prob_neigh = prob[neigh_node]
        lossnode = 0
        for k in prob_neigh:
            lossnode = lossnode + torch.linalg.norm(prob[i] - k)   ###norm square
        lossnode = lossnode - degs[i] * torch.linalg.norm(prob[i])  ###norm square
        loss_all.append(lossnode)

    loss_masked = torch.stack(loss_all)
    loss_masked = loss_masked[train_mask]
    loss_masked_ave = torch.mean(loss_masked)


    return loss_masked_ave


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    # training loop
    for epoch in range(200):
        model.train()
        logits = model(g, features)
        loss_ce = loss_fcn(logits[train_mask], labels[train_mask])
        loss_smooth = my_loss_mask(g,logits,train_mask,device='cuda')
        loss = loss_ce + 0.1* loss_smooth
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, loss.item(), acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    args = parser.parse_args()
    print(f'Training with DGL built-in GATConv module.')

    # load and preprocess dataset
    transform = AddSelfLoop()  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == 'cora':
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    test_acc=[]
    for _ in range(10):
        g = data[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        g = g.int().to(device)
        features = g.ndata['feat']
        labels = g.ndata['label']
        masks = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']

        # create GAT model
        in_size = features.shape[1]
        out_size = data.num_classes
        model = GAT(in_size, 8, out_size, heads=[8, 1]).to(device)

        # model training
        print('Training...')
        train(g, features, labels, masks, model)

        # test the model
        print('Testing...')
        acc = evaluate(g, features, labels, masks[2], model)
        print("Test accuracy {:.4f}".format(acc))
        print("*"*50)
        test_acc.append(acc)
    print("test_acc_clean: ", test_acc)
    print("test_acc_clean mean: ", statistics.mean(test_acc))
    print("test_acc_clean var: ", statistics.stdev(test_acc))
