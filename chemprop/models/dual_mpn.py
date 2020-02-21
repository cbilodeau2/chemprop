from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph, getHBonds
from chemprop.nn_utils import b_scope_tensor, batch_to_flat, flat_to_batch, index_select_ND, get_activation_function


class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.attn_dim = self.atom_fdim + self.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i1 = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        self.W_i2 = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        # W_h for projecting msgs
        # W_o for re-aggregating atomFeat+msg into atom hidden state
        self.W_h1 = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
        self.W_h2 = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o1 = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)
        self.W_o2 = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # Attention
        self.W_v1 = nn.Linear(self.attn_dim, self.hidden_size, bias=False)
        self.W_v2 = nn.Linear(self.attn_dim, self.hidden_size, bias=False)

    def localMsg(self, message, a2b, b2a, b2revb):
        if self.undirected:
            message = (message + message[b2revb]) / 2

        # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
        # message      a_message = sum(nei_a_message)      rev_message
        nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        rev_message = message[b2revb]  # num_bonds x hidden
        return a_message[b2a] - rev_message  # num_bonds x hidden

    def agg_bonds_to_atoms(self, passed_message, f_atoms, a2x):
        nei_a_message = index_select_ND(passed_message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        return torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)

    def readout(self, atom_hiddens, a_scope):
        """
        :param atom_hiddens: Hidden representation of size ?.
        """
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        return torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

    def forward(self, # TODO: refactor for consistency. mol <=> drug. struct <=> cmpd.
                mol_graph: BatchMolGraph,  # mols we attend to
                struct_graph: BatchMolGraph,  # structure we are reasoning about
                attn_coefs: torch.FloatTensor,  # attn coefs btwn graphs
                mol_features: List[np.ndarray] = None,
                struct_features: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features: # True if RDKit
            mol_features = torch.from_numpy(np.stack(mol_features)).float()
            struct_features = torch.from_numpy(np.stack(struct_features)).float()

            if self.args.cuda:
                mol_features = mol_features.cuda()
                struct_features = struct_features.cuda()

            if self.features_only:
                return (mol_features, struct_features)

        zero_vector = torch.zeros(1,self.hidden_size)
        mol_f_atoms, mol_f_bonds, mol_a2b, mol_b2a, mol_b2revb, mol_a_scope, _ = mol_graph.get_components()
        mol_b_scope, mol_b_revscope = b_scope_tensor(mol_a_scope)
        struct_f_atoms, struct_f_bonds, struct_a2b, struct_b2a, struct_b2revb, struct_a_scope, _ = struct_graph.get_components()
        struct_b_scope, struct_b_revscope = b_scope_tensor(struct_a_scope)

        if self.atom_messages:
            raise RuntimeError("Not supported yet!")
            a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            mol_f_atoms, mol_f_bonds, mol_a2b, mol_b2a, mol_b2revb = mol_f_atoms.cuda(), mol_f_bonds.cuda(), mol_a2b.cuda(), mol_b2a.cuda(), mol_b2revb.cuda()
            mol_b_scope, mol_b_revscope = mol_b_scope.cuda(), mol_b_revscope.cuda()
            struct_f_atoms, struct_f_bonds, struct_a2b, struct_b2a, struct_b2revb = struct_f_atoms.cuda(), struct_f_bonds.cuda(), struct_a2b.cuda(), struct_b2a.cuda(), struct_b2revb.cuda()
            struct_b_scope, struct_b_revscope = struct_b_scope.cuda(), struct_b_revscope.cuda()
            zero_vector = zero_vector.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()

        # Input
        if self.atom_messages:
            raise RuntimeError("Not supported yet!")
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            mol_input = self.W_i1(mol_f_bonds)  # num_bonds x hidden_size
            struct_input = self.W_i2(struct_f_bonds)  # num_bonds x hidden_size

        mol_message = self.act_func(mol_input)  # num_bonds x hidden_size
        struct_message = self.act_func(struct_input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            mol_message = self.localMsg(mol_message, mol_a2b, mol_b2a, mol_b2revb)  # num_bonds x hidden
            mol_message = self.act_func(mol_input + self.W_h1(mol_message))  # num_bonds x hidden_size
            mol_message = self.dropout_layer(mol_message)  # num_bonds x hidden

            struct_message = self.localMsg(struct_message, struct_a2b, struct_b2a, struct_b2revb)  # num_bonds x hidden
            struct_message = self.act_func(struct_input + self.W_h2(struct_message))  # num_bonds x hidden_size
            struct_message = self.dropout_layer(struct_message)  # num_bonds x hidden

        # Aggregate into hidden state of atoms
        a2x = mol_a2a if self.atom_messages else mol_a2b
        mol_a_input = self.agg_bonds_to_atoms(mol_message, mol_f_atoms, a2x)
        a2x = struct_a2a if self.atom_messages else struct_a2b
        struct_a_input = self.agg_bonds_to_atoms(struct_message, struct_f_atoms, a2x)

        # Project into value space for attn summing
        mol_val = self.W_v1(mol_a_input)
        struct_val = self.W_v2(struct_a_input)
        mol_val = flat_to_batch(mol_val, mol_b_scope)  # reshape into batches
        struct_val = flat_to_batch(struct_val, struct_b_scope)  # reshape into batches

        # sum scores * msg of mol
        struct_scores = F.softmax(attn_coefs, dim=1)
        mol_add_attn = torch.matmul(struct_scores.unsqueeze(-1), struct_val.unsqueeze(2))
        mol_add_attn = mol_add_attn.sum(dim=1)
        mol_add_attn = batch_to_flat(mol_add_attn, mol_b_revscope)

        mol_scores = F.softmax(torch.transpose(attn_coefs, dim0=1, dim1=2), dim=1)
        struct_add_attn = torch.matmul(mol_scores.unsqueeze(-1), mol_val.unsqueeze(2))
        struct_add_attn = struct_add_attn.sum(dim=1)
        struct_add_attn = batch_to_flat(struct_add_attn, struct_b_revscope)

        # Add dummy row
        mol_val = torch.cat((zero_vector, mol_add_attn), dim=0)
        struct_val = torch.cat((zero_vector, struct_add_attn), dim=0)

        # Incorporate attn into atom hidden msgs
        mol_hiddens = self.act_func(self.W_o1(mol_a_input) + mol_val)  # num_atoms x hidden
        struct_hiddens = self.act_func(self.W_o2(struct_a_input) + struct_val)  # num_atoms x hidden
        mol_hiddens = self.dropout_layer(mol_hiddens)  # num_atoms x hidden
        struct_hiddens = self.dropout_layer(struct_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = self.readout(mol_hiddens, mol_a_scope)
        struct_vecs = self.readout(struct_hiddens, struct_a_scope)

        if self.use_input_features:  # True if features_path or features_generator specified
            mol_features = mol_features.to(mol_vecs)
            struct_features = struct_features.to(struct_vecs)
            if len(mol_features.shape) == 1:
                mol_features = mol_features.view([1,mol_features.shape[0]])
                struct_features = struct_features.view([1,struct_features.shape[0]])
            mol_vecs = torch.cat([mol_vecs, mol_features], dim=1)  # (num_molecules, hidden_size)
            struct_vecs = torch.cat([struct_vecs, struct_features], dim=1)  # (num_molecules, hidden_size)

        return mol_vecs, struct_vecs  # num_molecules x hidden


class DualMPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(DualMPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def get_attn_coefs(self,
            drug_batch: Union[List[str], BatchMolGraph],
            cmpd_batch: Union[List[str], BatchMolGraph]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        drug_batch = mol2graph(drug_batch, self.args)
        cmpd_batch = mol2graph(cmpd_batch, self.args)
        attn_coefs = getHbonds(drug_batch, cmpd_batch)

        struct_scores = F.softmax(attn_coefs, dim=1)
        mol_scores = F.softmax(torch.transpose(attn_coefs, dim0=1, dim1=2), dim=1)
        return mol_scores, struct_scores

    def forward(self,
                drug_batch: Union[List[str], BatchMolGraph],
                cmpd_batch: Union[List[str], BatchMolGraph],
                drug_feats: List[np.ndarray] = None,
                cmpd_feats: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param drug_batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param cmpd_batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param drug_feats: A list of ndarrays containing additional features.
        :param cmpd_feats: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            drug_batch = mol2graph(drug_batch, self.args)
            cmpd_batch = mol2graph(cmpd_batch, self.args)
            attn_coefs = getHbonds(drug_batch, cmpd_batch)

        output = self.encoder.forward(drug_batch, cmpd_batch, attn_coefs, drug_feats, cmpd_feats)

        return output
