from argparse import Namespace

import torch
import torch.nn as nn

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: Namespace):
        """
        Initializes the MoleculeModel.

        :param args: Arguments.
        """
        super(MoleculeModel, self).__init__()
        self.create_encoder(args)
        self.create_ffn(args)
        initialize_weights(self)
        self.create_contexts(args)  # Has uniform initialization rather than Xavier

        self.n_contexts = args.n_contexts
        self.hidden_size = args.hidden_size
        self.softmax = nn.Softmax(dim=1)
        # self.activation = nn.Identity()
        # if args.output_raw:
        self.activation = nn.Sigmoid()

    def create_encoder(self, args: Namespace):
        """
        Creates the paired message passing encoders for the model.

        :param args: Arguments.
        """
        self.drug_encoder = MPN(args)
        self.cmpd_encoder = MPN(args)

    def create_contexts(self, args: Namespace):
        self.contexts = nn.Embedding(args.n_contexts, args.hidden_size)
        self.contexts.weight.data.uniform_(-0.001, 0.001)

        self.index = torch.as_tensor([list(range(args.n_contexts))], dtype=torch.long)
        if args.cuda:
            self.index = self.index.cuda()

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        self.ops = args.ops

        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size*2
            if args.use_input_features:
                first_linear_dim += args.features_dim

        if args.drug_only or args.cmpd_only or self.ops != 'concat':
            first_linear_dim = int(first_linear_dim/2)

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        smiles, _ = input  # smiles looks like [(d1,c1), (d2,c2), ...]
        n_mols = len(smiles)
        drugs, cmpds = zip(*smiles)

        drug_embeds = self.drug_encoder(drugs).squeeze(-1)
        context_embeds = self.contexts(self.index)
        distr = torch.matmul(drug_embeds, context_embeds.squeeze(0).T)
        distr = self.softmax(distr)

        cmpd_embeds = self.cmpd_encoder(cmpds)
        cmpd_embeds = cmpd_embeds.unsqueeze(1).expand([n_mols, self.n_contexts, self.hidden_size])
        context_embeds = context_embeds.expand([n_mols, self.n_contexts, self.hidden_size])

        newInput = [context_embeds, cmpd_embeds]
        if self.ops == 'plus':
            newInput = newInput[0] + newInput[1]
        elif self.ops == 'minus':
            newInput = newInput[0] - newInput[1]
        else:
            newInput = torch.cat(newInput, dim=-1)

        output = self.ffn(newInput)
        output = self.activation(output)
        print(output[0])
        output = torch.bmm(distr.unsqueeze(1), output)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if not self.training:
            output = self.activation(output)

        return output.squeeze(-1)


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size  # Redundant bc of multiclass
    if args.dataset_type == 'multiclass':
        raise NotImplementedError

    model = MoleculeModel(args)

    return model
