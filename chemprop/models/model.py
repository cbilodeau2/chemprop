from argparse import Namespace
from typing import Dict, List, Union

import torch
import torch.nn as nn

from .mpn import MPN
from .embed import Embedding
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool, mse: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.softmax=nn.LogSoftmax(dim=0)
        assert (classification and not multiclass)

    def init_embeddings(self, args: Namespace,
            drug_set: Union[Dict[str, int], List[str]],
            cmpd_set: Union[Dict[str, int], List[str]]):
        """
        Creates and initializes the independent embedding layer for the model.

        :param args: Arguments.
        :param drug_set: Set of unique drug compounds.
        :param cmpd_set: Set of unique cmpd compounds.
        """
        if type(drug_set) == list:
            drug_set = {x: i for i, x in enumerate(drug_set)}
            cmpd_set = {x: i for i, x in enumerate(cmpd_set)}
        assert not args.shared
        self.shared = False

        self.drug_encoder = Embedding(args, drug_set)
        self.cmpd_encoder = Embedding(args, cmpd_set)

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
                ret.append(self.softmax(output[start:start+size])[0])  # only the pos one
                start += size
            ret = torch.cat(ret)
            return ret

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
    if args.dataset_type == 'multiclass':
        raise NotImplementedError

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass', mse=args.loss_func == 'mse')

    if args.embedding:
        model.init_embeddings(args, drug_set, cmpd_set)
    else:
        model.create_encoder(args)
        initialize_weights(model)  # initialize xavier for both

    return model
