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

        self.logsoftmax=nn.LogSoftmax(dim=0)
        assert (classification and not multiclass)

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
        smiles, lengths = input

        learned_drug = self.drug_encoder([x[0] for x in smiles])
        learned_drug = learned_drug.unsqueeze(1)
        if self.shared:
            learned_cmpd = self.drug_encoder([x[0] for x in smiles])
        else:
            learned_cmpd = self.cmpd_encoder([x[1] for x in smiles])
        learned_cmpd = learned_cmpd.unsqueeze(-1)

        output = torch.bmm(learned_drug, learned_cmpd).squeeze(-1)

        if self.training:
            start, ret = 0, []
            for size in lengths:
                ret.append(self.logsoftmax(output[start:start+size])[0])  # only the pos one
                start += size
            ret = torch.cat(ret)
            return ret

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
        raise NotImplementedError

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
