import sys
import pickle as pkl
from collections import OrderedDict

from embedMatrix import MatrixEst
from chemprop.models import MPN
from chemprop.parsing import parse_train_args
from chemprop.nn_utils import initialize_weights

import torch
import torch.nn as nn
import torch.optim as optim
sys.argv = ['train.py', '--data_path', 'tmp.csv', '--dataset_type', 'classification', '--num_folds', '10', '--metric', 'prc-auc', '--embed_dir', 'tmp.lol']
args = parse_train_args()

MODEL_DIR = 'aggModels_dim300_batch5k/'

with open(MODEL_DIR + 'embedMap.pkl','rb') as f:
    idx_map = pkl.load(f)
    drugIdx = idx_map['drug']
    cmpdIdx = idx_map['cmpd']

rev_drugIdx = {drugIdx[key]: key for key in drugIdx}
rev_cmpdIdx = {cmpdIdx[key]: key for key in cmpdIdx}

drug_inp = [rev_drugIdx[i] for i in range(len(drugIdx))]
cmpd_inp = [rev_cmpdIdx[i] for i in range(len(cmpdIdx))]
# embed = MatrixEst(len(drug_idx), len(cmpd_idx))
embed = torch.load(MODEL_DIR + 'fold_0.pt')

gcn = MPN(args)
initialize_weights(gcn)

optimizer = optim.Adam(gcn.parameters(),lr=1e-3)
criterion = nn.MSELoss()
target = embed['drug.weight']

gcn.train()
for epoch in range(30):
    optimizer.zero_grad()
    output = gcn(drug_inp)
    loss = criterion(output, target)

    loss.backward()
    optimizer.step()
    if epoch%2 == 0:
        print(epoch, float(loss)/len(drug_inp))
print('Compared to norm', target.norm(dim=1).mean())
torch.save(gcn.state_dict(), 'drug_gcn2.pt')
