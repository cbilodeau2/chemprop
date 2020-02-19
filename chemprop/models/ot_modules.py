import torch
import ot
import os
import numpy as np
import random


def compute_cost_mat(X_1, X_2, dist_type='l2', rescale_cost=True):
    '''Computes the l2 cost matrix between two point cloud inputs.

    Args:
        X_1: [#nodes_1, #features] point cloud, tensor
        X_2: [#nodes_2, #features] point cloud, tensor
        rescale_cost: Whether to normalize the cost matrix by the max ele

    Output:
        [#nodes_1, #nodes_2] matrix of the l2 distance between point pairs
    '''
    if dist_type == 'l2':
        n_1, _ = X_1.shape
        n_2, _ = X_2.shape

        # Expand dimensions to broadcast in the following step
        X_1 = X_1.view(n_1, 1, -1)
        X_2 = X_2.view(1, n_2, -1)
        squared_dist = (X_1 - X_2) ** 2
        cost_mat = torch.sum(squared_dist, dim=2)

    elif dist_type == 'dot':
        cost_mat = - X_1.matmul(X_2.transpose(0,1))

    else:
        raise NotImplementedError('Unsupported dist type')

    if rescale_cost:
        cost_mat = cost_mat / cost_mat.max()

    return cost_mat


def compute_ot(X_1, X_2, cuda, dist_type='l2', opt_method='emd', rescale_cost=False):
    ''' Computes the optimal transport distance

    Args:
        X_1: [#nodes_1, #features] point cloud, tensor
        X_2: [#nodes_2, #features] point cloud, tensor
        H_1: [#nodes_1] histogram, numpy array
        H_2: [#nodes_2] histogram, numpy array
        dist_type: 'l2' or 'dot'
        cuda: bool indiciating if gpu should be used
        opt_method: The optimization method {emd, wmd}
        rescale_cost: Whether to normalize the cost matrix by the max ele. Get errors if you do this
    '''
    drug_numAtoms, cmpd_numAtoms = X_1.shape[0], X_2.shape[0]
    H_1 = np.ones(drug_numAtoms)/drug_numAtoms
    H_2 = np.ones(cmpd_numAtoms)/cmpd_numAtoms
    # raise ValueError(X_1.shape, X_2.shape, H_1.shape, H_2.shape)

    cost_mat = compute_cost_mat(X_1, X_2, dist_type, rescale_cost=rescale_cost)
    # Convert cost matrix to numpy array to feed into sinkhorn algorithm
    cost_mat_detach = cost_mat.detach().cpu().numpy()
    if opt_method == 'emd':
        M = np.max(np.abs(cost_mat_detach)) + cost_mat_detach
        ot_mat = ot.emd(a=H_1, b=H_2, M=M, numItermax=10000)

    elif opt_method == 'wmd':
        ot_dist = torch.max(torch.mean(torch.min(cost_mat, dim=0)[0]), torch.mean(torch.min(cost_mat, dim=1)[0]))
        return ot_dist, None

    # if random.random() < 0.05 and abs(np.sum(ot_mat).item() - 1.) > 1e-2:
        # print('ot mat' , ot_mat)
        # print('positive cost mat', np.max(np.abs(cost_mat_detach)) + cost_mat_detach)
        # print('cost mat', cost_mat_detach)
        # print('x1', X_1)
        # print('x2', X_2)
        # print('h1', H_1)
        # print('h2', H_2)
        # print('sum rows', ot_mat.sum(1) * H_1.shape[0])
        # print('sum cols', ot_mat.sum(0) * H_2.shape[0])
        # assert abs(np.sum(ot_mat).item() - 1.) < 1e-2

    ot_mat_attached = torch.tensor(ot_mat, device='cuda' if cuda else 'cpu', requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)

    return ot_dist, ot_mat_attached
