import logging
from typing import Callable, List

import torch.nn as nn

from .predict import predict
from chemprop.data import StratMolPairDataset, MolPairDataset, StandardScaler


def val_loss(model: nn.Module,
             data: Union[StratMolPairDataset, List[StratMolPairDataset]],
             loss_func: Callable,
             batch_size: int) -> int:
    """
    Gets validation loss for an epoch.

    :param model: Model.
    :param data: A MolPairDataset (or a list of MolPairDatasets if using moe).
    :param loss_func: Loss function.
    :param batch_size: Batch size.
    :return: loss on validation set.
    """
    model.train()
    data.shuffle()
    loss_sum, num_pos = 0, 0

    for i in trange(0, len(data), batch_size):
        mol_batch, lengths = data[i:i + batch_size]
        num_pos += len(lengths)
        mol_batch = MolPairDataset(mol_batch)

        batch = mol_batch.smiles()

        # Run model
        model.zero_grad()
        with torch.no_grad():
            preds = model(batch, lengths)
            ind_select = torch.zeros(preds.shape, dtype=torch.long)
            if next(model.parameters()).is_cuda:
                ind_select = ind_select.cuda()

            # Output only the positive ones
            loss = loss_func(preds.unsqueeze(-1), ind_select).sum()
            loss_sum += loss.item()

    return loss_sum/num_pos

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
             data: StratMolPairDataset,
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
    loss = val_loss(model, data, loss_func, batch_size)

    data = data.getMolPairDataset()
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
