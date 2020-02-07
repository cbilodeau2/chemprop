from argparse import Namespace

import torch
import torch.nn as nn

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace):
        """
        Creates the paired message passing encoders for the model.

        :param args: Arguments.
        """
        self.shared = args.shared
        self.drug_encoder = MPN(args)
        if not self.shared:
            self.cmpd_encoder = MPN(args)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        smiles, feats = input

        learned_drug = self.drug_encoder([x[0] for x in smiles], [x[0] for x in feats])
        learned_drug = learned_drug.unsqueeze(1)
        if self.shared:
            learned_cmpd = self.drug_encoder([x[0] for x in smiles], [x[0] for x in feats])
        else:
            learned_cmpd = self.cmpd_encoder([x[1] for x in smiles], [x[1] for x in feats])
        learned_cmpd = learned_cmpd.unsqueeze(-1)

        output = torch.bmm(learned_drug, learned_cmpd).squeeze(-1)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training and args.loss_func != 'mse':
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')
    model.create_encoder(args)

    initialize_weights(model)

    return model
