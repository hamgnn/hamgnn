import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

import torch.nn as nn
# from torchdiffeq import odeint_adjoint as odeint
# from torchdiffeq import odeint as odeint
import os
import pickle
import numpy as np

# model = torch.load('./experiment3/model_corav5extend2ncmodelgeo5.pth')
#
#
# with open('data_corageo5.pkl','rb') as f:
#     data = pickle.load(f)
#

model = torch.load('./experiment3/model_corav5extend0ncGeoGCN11model.pth')


with open('data_coraGeoGCN11.pkl','rb') as f:
    data = pickle.load(f)

x_raw = data['features']
adj = data['adj_train_norm']
model.eval()
model.to('cpu')
with torch.no_grad():
    output = model.encode(x_raw,adj)

print(output.shape)
# cos = nn.CosineSimilarity(dim=1, eps=1e-6)
# out_cos = cos(output, output)
# print(out_cos.numpy().mean())
# print(out_cos.shape)
#
# from scipy.spatial.distance import pdist,cdist
#
# out_dis = pdist(output,metric='cosine')
#
# print(out_dis.shape)


from scipy.spatial.distance import pdist,cdist
out_cdis = cdist(output, output, metric='cosine')
print(out_cdis.shape)

average_row =out_cdis.sum(1)/(out_cdis!=0).sum(1)

print(average_row.shape)
cos_score = average_row[np.nonzero(average_row)].mean()
print(cos_score)

