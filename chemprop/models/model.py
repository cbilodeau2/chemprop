from argparse import Namespace
from typing import Dict, List, Union

import torch
import torch.nn as nn

from .mpn import MPN
from .ot_modules import compute_ot
from chemprop.nn_utils import initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, dataset_type: str, dist: str, mse_loss: bool, gpu: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.gpu = gpu
        self.classification = dataset_type == 'classification'
        self.regression = dataset_type == 'regression'
        if self.classification:
            self.activation = nn.Identity() if mse_loss else nn.Sigmoid()
        elif self.regression:
            self.scale = nn.Linear(1, 1, bias=True)
        self.dist = dist

    def create_encoder(self, args: Namespace):
        """
        Creates the paired message passing encoders for the model.

        :param args: Arguments.
        """
        self.drug_encoder = MPN(args)
        if args.shared:
            self.cmpd_encoder = self.drug_encoder
        else:
            self.cmpd_encoder = MPN(args)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        smiles, feats = input

        learned_drugs = self.drug_encoder([x[0] for x in smiles], [x[0] for x in feats])
        learned_cmpds = self.cmpd_encoder([x[1] for x in smiles], [x[1] for x in feats])
        assert len(learned_drugs) == len(learned_cmpds)

        output = []
        for i, drug in enumerate(learned_drugs):
            cmpd = learned_cmpds[i]
            dist, _ = compute_ot(drug, cmpd, self.gpu, opt_method=self.dist)
            output.append(dist)
        output = torch.stack(output, dim=0).unsqueeze(-1)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification:
            output = -output
        if self.classification and not self.training:  # is identity if mse loss
            output = self.activation(output)
        if self.regression:
            output = self.scale(output)

        return output


def build_model(args: Namespace,
        drug_set: Union[Dict[str, int], List[str]] = None,
        cmpd_set: Union[Dict[str, int], List[str]] = None) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers. If smiles sets are provided, then independent embeddings replace the MPNN.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size

    model = MoleculeModel(dataset_type=args.dataset_type, dist=args.dist, mse_loss=args.loss_func == 'mse', gpu=args.cuda)
    model.create_encoder(args)
    initialize_weights(model)  # initialize xavier for both

    return model
