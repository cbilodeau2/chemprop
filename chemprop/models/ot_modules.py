import torch
import ot
import os
import numpy as np
import random


def compute_cost_mat(X_1, X_2, dist_type='l2', rescale_cost):
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
        if rescale_cost:
            cost_mat = cost_mat / cost_mat.max()

    elif dist_type == 'dot':
        cost_mat = - X_1.matmul(X_2.transpose(0,1))
        if rescale_cost:
            cost_mat = cost_mat / abs(cost_mat.min())

    else:
        raise NotImplementedError('Unsupported dist type')

    return cost_mat


def compute_ot(X_1, X_2, cuda, dist_type, opt_method='emd', rescale_cost=False):
    ''' Computes the optimal transport distance

    Args:
        X_1: [#nodes_1, #features] point cloud, tensor
        X_2: [#nodes_2, #features] point cloud, tensor
        cuda: bool indiciating if gpu should be used
        dist_type: Distance for cost matrix. {l2, dot}
        opt_method: The optimization method {emd, wmd}
        rescale_cost: Whether to normalize the cost matrix to be btwn [0,1].
    '''
    drug_numAtoms, cmpd_numAtoms = X_1.shape[0], X_2.shape[0]
    H_1 = np.ones(drug_numAtoms)/drug_numAtoms
    H_2 = np.ones(cmpd_numAtoms)/cmpd_numAtoms

    cost_mat = compute_cost_mat(X_1, X_2, dist_type, rescale_cost=rescale_cost)
    # Convert cost matrix to numpy array to feed into sinkhorn algorithm
    cost_mat_detach = cost_mat.detach().cpu().numpy()
    if opt_method == 'emd':
        M = np.max(np.abs(cost_mat_detach)) + cost_mat_detach
        ot_mat = ot.emd(a=H_1, b=H_2, M=M, numItermax=10000)

    elif opt_method == 'wmd':
        ot_dist = torch.max(torch.mean(torch.min(cost_mat, dim=0)[0]), torch.mean(torch.min(cost_mat, dim=1)[0]))
        return ot_dist, None, torch.Tensor([0])

    ot_mat_attached = torch.tensor(ot_mat, device='cuda' if cuda else 'cpu', requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)

    all_nce_dists = []
    all_nce_dists.append(-ot_dist)
    for _ in range(5):
        random_mat = ot_mat.copy()
        np.random.shuffle(random_mat)
        random_mat = torch.tensor(random_mat, device='cuda' if cuda else 'cpu', requires_grad=False).float()
        all_nce_dists.append(-torch.sum(random_mat * cost_mat))

    for _ in range(5):
        random_mat = np.random.rand(H_1.shape[0], H_2.shape[0]) * 10.
        while np.linalg.norm(np.sum(random_mat, axis=0) - H_2) > 1e-3 and\
                np.linalg.norm(np.sum(random_mat, axis=1) - H_1) > 1e-3:
            random_mat = random_mat / np.sum(random_mat, axis=0, keepdims=True) * H_2.reshape((1, H_2.shape[0]))
            random_mat = random_mat / np.sum(random_mat, axis=1, keepdims=True) * H_1.reshape((H_1.shape[0], 1))
        random_mat = torch.tensor(random_mat, device='cuda' if cuda else 'cpu', requires_grad=False).float()
        all_nce_dists.append(-torch.sum(random_mat * cost_mat))

    nce_reg = torch.nn.LogSoftmax(dim=0)(torch.stack(all_nce_dists))[0]

    return ot_dist, ot_mat_attached, nce_reg
