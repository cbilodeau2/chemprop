from argparse import Namespace
import numpy as np
from typing import Dict, List, Union

from rdkit import Chem
from .maccsKeys import smartsPatts as maccs

"""
Co-opt from https://github.com/wengong-jin/chemprop/blob/master/chemprop/features/functional_groups.py
"""

class FunctionalGroupFeaturizer:
    """
    Class for extracting feature vector of indicators for atoms being parts of functional groups in Maccs fingerprints.
    """

    def __init__(self):
        self.smarts = []
        for key in maccs.keys():
            patt, _ = maccs[key]
            if patt != '?':
              self.smarts.append(Chem.MolFromSmarts(patt))

    def get_dim(self):
        return len(self.smarts)

    def featurize(self, smiles: Union[Chem.Mol, str]) -> List[List[int]]:
        """
        Given a molecule in SMILES form, return a feature vector of indicators for each atom,
        indicating whether the atom is part of each functional group. 
        Can also directly accept a Chem molecule.
        Searches through the functional groups given in smarts.txt. 
        :param smiles: A smiles string representing a molecule.
        :return: Numpy array of shape num_atoms x num_features (num functional groups)
        """
        if type(smiles) == str:
            mol = Chem.MolFromSmiles(smiles)  # turns out rdkit knows to match even without adding Hs
        else:
            mol = smiles
        features = np.zeros((mol.GetNumAtoms(), len(self.smarts)))  # num atoms (without Hs) x num features
        for i, smarts in enumerate(self.smarts):
            for group in mol.GetSubstructMatches(smarts):
                for idx in group:
                    features[idx][i] = 1

        return features


if __name__ == '__main__':
    featurizer = FunctionalGroupFeaturizer()
    features = featurizer.featurize('C(#N)C(=O)C#N')
    print(np.array(features).shape)
