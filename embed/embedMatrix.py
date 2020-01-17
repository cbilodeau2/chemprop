from collections import defaultdict

import os
import numpy as np
import pandas as pd
import pickle as pkl
import sklearn.metrics as skm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dutils
import torch.optim as optim

def make_dir(directory):
    if not os.path.exists(directory):
        print('Created', directory)
        os.makedirs(directory)

def get_data():
    data = pd.read_csv(DATA_PATH)
    data.columns = ['drug', 'cmpd', 'prop'] # + list(data.columns[2:])

    drug_idxs = {drug: i for i, drug in enumerate(set(data['drug']))}
    cmpd_idxs = {cmpd: i for i, cmpd in enumerate(set(data['cmpd']))}
    map_path = MODEL_PATH + 'embedMap.pkl'
    with open(map_path, 'wb') as f:
        pkl.dump({'drug': drug_idxs, 'cmpd': cmpd_idxs}, f)

    data['drug_idx'] = data['drug'].map(drug_idxs)
    data['cmpd_idx'] = data['cmpd'].map(cmpd_idxs)

    return data, len(drug_idxs), len(cmpd_idxs)

def split_data(data, fold_num):
    flag = False
    splits = pkl.load(open(FOLD_PATH + 'fold_{}.pkl'.format(fold_num), 'rb'))
    train_data, val_data, test_data = [data.iloc[splits[i]] for i in range(3)]

    if invalid_split(test_data):
        print('fold {}: test split INVALID'.format(num_run))
        flag = True
    if invalid_split(val_data):
        print('fold {}: val split INVALID'.format(num_run))
        flag = True

    # Convert cmpds to embedding refs
    train_idxs = torch.tensor([list(train_data['drug_idx']), list(train_data['cmpd_idx'])], dtype=torch.long).T
    val_idxs = torch.tensor([list(val_data['drug_idx']), list(val_data['cmpd_idx'])], dtype=torch.long).T
    test_idxs = torch.tensor([list(test_data['drug_idx']), list(test_data['cmpd_idx'])], dtype=torch.long).T

    # Convert truth to tensors
    train_class = torch.tensor([list(train_data['prop'])], dtype=torch.float).T
    val_class = torch.tensor([list(val_data['prop'])], dtype=torch.float).T
    test_class = torch.tensor([list(test_data['prop'])], dtype=torch.float).T

    # for hinge loss
    # train_class = 2*(train_class-0.5)
    # val_class = 2*(val_class-0.5)
    # test_class = 2*(test_class-0.5)

    # Create dataset
    train = dutils.TensorDataset(train_idxs, train_class)
    val = dutils.TensorDataset(val_idxs, val_class)
    test = dutils.TensorDataset(test_idxs, test_class)

    return train, val, test, flag

def invalid_split(split):
    return split['prop'].value_counts().get(1) is None

def prc_auc(targets, preds):
    """
    Computes the area under the precision-recall curve.
    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = skm.precision_recall_curve(targets, preds)
    return skm.auc(recall, precision)

def train_net(net, loaders, save_path=None, epochs=30, verbose=False):
    train_loader, val_loader, test_loader = loaders
    # create your optimizer
    optimizer = optim.Adam(net.parameters(),lr=1e-2)
    # criterion = nn.MSELoss()
    # criterion = nn.HingeEmbeddingLoss()
    # USE w/ hinge loss target = 2*(target-0.5)
    criterion = nn.BCEWithLogitsLoss() # chemprop uses mean

    trainLosses = []
    valLosses = []
    testLosses = []

    bestValScore = -float('inf')
    valMetrics = defaultdict(list)
    testMetrics = defaultdict(list)

    for epoch in range(epochs):
        train_loss = 0
        net.train()
        for i, (input, target) in enumerate(train_loader):
            optimizer.zero_grad()  # zero the gradient buffers
            output = net(input)
            loss = criterion(output.view(target.shape), target)

            loss.backward()
            optimizer.step()  # Does the update
            train_loss += loss.item()
        trainLosses.append(train_loss/len(train_loader))

        net.eval()
        val_loss = 0
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for input, target in val_loader:
                output = net(input)
                output = torch.sigmoid(output)  # if BCELogits used
                all_targets.append(target.numpy())
                all_preds.append(output.numpy().squeeze(1))

                loss = criterion(output.view(target.shape), target)
                val_loss += loss.item()
        valLosses.append(val_loss/len(val_loader))
        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)
        valMetrics['auc'].append( skm.roc_auc_score(all_targets, all_preds) )
        valMetrics['prc'].append( prc_auc(all_targets, all_preds) )

        test_loss = 0
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for input, target in test_loader:
                output = net(input)
                output = torch.sigmoid(output) # if BCELogits used

                all_targets.append(target.numpy())
                all_preds.append(output.numpy().squeeze(1))

                loss = criterion(output.view(target.shape), target)
                test_loss += loss.item()
        testLosses.append(test_loss/len(test_loader))
        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)
        testMetrics['auc'].append( skm.roc_auc_score(all_targets, all_preds) )
        testMetrics['prc'].append( prc_auc(all_targets, all_preds) )

        # Save checkpoints
        if save_path and valMetrics[METRIC][-1] > bestValScore:
            torch.save(net.state_dict(), save_path)

        if verbose and epoch%10==0:
            print('epoch {0}: {1:.6f} (trainLoss); {2:.6f} (valLoss); {3:.6f} (testLoss)'
                    .format(epoch, trainLosses[-1], valLosses[-1], testLosses[-1]))
    return valMetrics, testMetrics

class MatrixEst(nn.Module):
    def __init__(self, drug_size, cmpd_size, hidden_size=300):
        super(MatrixEst, self).__init__()
        self.drug = nn.Embedding(drug_size, hidden_size)
        # nn.init.xavier_normal_(self.drug.weight)
        lol = 1/drug_size
        self.drug.weight.data.uniform_(-lol, lol)
        self.cmpd = nn.Embedding(cmpd_size, hidden_size)
        # nn.init.xavier_normal_(self.cmpd.weight)
        lol = 1/cmpd_size
        self.cmpd.weight.data.uniform_(-lol, lol)

        ffn = [
                nn.Linear(hidden_size*2, 300, bias=True),
                nn.ReLU(),
                nn.Linear(300, 1, bias=True)
                ]
        self.ffn = nn.Sequential(*ffn)

    def forwardFfn(self, inputs):
        drug = self.drug(inputs[:, 0])
        cmpd = self.cmpd(inputs[:, 1])
        return self.ffn(torch.cat([drug, cmpd], dim=1)).unsqueeze(1)

    def forward(self, inputs):
        drug = self.drug(inputs[:, 0]).unsqueeze(2)
        cmpd = self.cmpd(inputs[:, 1]).unsqueeze(2)
        cmpd = cmpd.permute(0,2,1)
        return torch.bmm(cmpd, drug) # if BCELogits used
        # return torch.sigmoid(torch.bmm(cmpd, drug)) # if BCELogits not used

"""
END OF UTILITIES
"""

DATA_PATH = 'agg.csv'
MODEL_PATH = 'aggModels_dim300_batch5k_initXavier_prc/'
FOLD_PATH = 'aggRandSplits/'
HIDDEN_SIZE = 300
METRIC = 'prc'

if __name__ == '__main__':
    make_dir(MODEL_PATH)
    data, drug_size, cmpd_size = get_data()
    perf = defaultdict(list)
    for num_run in range(10):
        save_path = MODEL_PATH + 'fold_{}.pt'.format(num_run)
        print('Starting run', num_run, 'saving to', save_path)
        train, val, test, invalid = split_data(data, num_run)
        if invalid:
            print('Run', num_run, 'early exit')
            continue
        train_loader = dutils.DataLoader(train, shuffle=True, batch_size=5000)
        val_loader = dutils.DataLoader(val, shuffle=True, batch_size=5000)
        test_loader = dutils.DataLoader(test, shuffle=True, batch_size=5000)

        net = MatrixEst(drug_size, cmpd_size, HIDDEN_SIZE)
        valScore, testScore = train_net(net, (train_loader, val_loader, test_loader), save_path, verbose=True)

        for key in valScore:
            ind = np.argmax(valScore[key])
            perf[key].append(testScore[key][ind])
            print('Best {3} from epoch {0}: {1:.6f} (val); {2:.6f} (test)'
                    .format(ind, valScore[key][ind], testScore[key][ind], key))

    with open(MODEL_PATH+'res.txt','a') as f:
        print("=== Key summary ===")
        print("=== Key summary ===", file=f)
        for key in perf:
            print(key, np.mean(perf[key]), '+/-', np.std(perf[key]))
            print(key, np.mean(perf[key]), '+/-', np.std(perf[key]), file=f)
