import logging
from typing import Callable, List, Union

import torch
import torch.nn as nn

from .loss_funcs import ContrastiveLoss
from .predict import predict
from chemprop.data import  convert2contrast, MolPairDataset, StandardScaler


def val_loss(model: nn.Module,
             data: Union[MolPairDataset, List[MolPairDataset]],
             loss_func: Callable,
             batch_size: int,
             dataset_type: str,
             scaler: StandardScaler = None) -> int:
    """
    Gets validation loss for an epoch.
    :param model: Model.
    :param data: A MolPairDataset (or a list of MolPairDatasets if using more).
    :param loss_func: Loss function.
    :param batch_size: Batch size.
    :param dataset_type: Dataset type.
    :param scaler: Scaler for data.
    :return: loss on validation set.
    """
    model.train()
    data.shuffle()
    if type(loss_func) == ContrastiveLoss:
        data = convert2contrast(data)
    loss_sum, total_num = 0, 0

    for i in range(0, len(data), batch_size):
        mol_batch = MolPairDataset(data[i:i + batch_size])
        smiles_batch, fractions_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.fractions(), mol_batch.features(), mol_batch.targets()
        # TODO: Apply scaling to features

        # Apply inverse scaling if regression
        if scaler is not None:
            target_batch = scaler.transform(target_batch)

        batch = smiles_batch
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])
        if type(loss_func) == ContrastiveLoss:
            mask = targets
        else:
            mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])

        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()

        # Run model
        model.zero_grad()
        with torch.no_grad():
            preds = model(batch, fractions_batch, features_batch)

            if dataset_type == 'multiclass':
                targets = targets.long()
                loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * mask
            else:
                loss = loss_func(preds, targets) * mask
            loss_sum += loss.sum().item()
            total_num += mask.sum()

    return loss_sum/total_num


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metric_func: Callable,
                         dataset_type: str,
                         logger: logging.Logger = None) -> List[float]:
    """
    Evaluates predictions using a metric function and filtering out invalid targets.

    :param preds: A list of lists of shape (data_size, num_tasks) with model predictions.
    :param targets: A list of lists of shape (data_size, num_tasks) with targets.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param dataset_type: Dataset type.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    info = logger.info if logger is not None else print

    if len(preds) == 0:
        return [float('nan')] * num_tasks

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = []
    for i in range(num_tasks):
        # # Skip if all targets or preds are identical, otherwise we'll crash during classification
        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                info('Warning: Found a task with targets all 0s or all 1s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                info('Warning: Found a task with predictions all 0s or all 1s')

            if nan:
                results.append(float('nan'))
                continue

        if len(valid_targets[i]) == 0:
            continue

        if dataset_type == 'multiclass':
            results.append(metric_func(valid_targets[i], valid_preds[i], labels=list(range(len(valid_preds[i][0])))))
        else:
            results.append(metric_func(valid_targets[i], valid_preds[i]))

    return results


def evaluate(model: nn.Module,
             data: MolPairDataset,
             loss_func: Callable,
             num_tasks: int,
             metric_func: Callable,
             batch_size: int,
             dataset_type: str,
             scaler: StandardScaler = None,
             logger: logging.Logger = None) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MolPairDataset.
    :param loss_func: Loss function.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param batch_size: Batch size.
    :param dataset_type: Dataset type.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    loss = val_loss(model, data, loss_func, batch_size, dataset_type, scaler)

    preds = predict(
        model=model,
        data=data,
        batch_size=batch_size,
        scaler=scaler
    )

    targets = data.targets()

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=dataset_type,
        logger=logger
    )

    return results, loss
