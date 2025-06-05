from sklearn.base import BaseEstimator, ClusterMixin
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
from collections import defaultdict


class ScaffoldClustering(BaseEstimator, ClusterMixin):
    def __init__(self):
        self.labels_ = None
        self.clusters_ = None
        self.scaffolds_ = None
        self.X_ = None

    def _validate_inputs(self, X):
        if not isinstance(X, (list, tuple)):
            raise TypeError("Input must be a list or tuple of RDKit molecules.")
        for i, mol in enumerate(X):
            if mol is None or not isinstance(mol, Chem.Mol):
                raise ValueError(f"Element at index {i} is not a valid RDKit Mol object.")

    def fit(self, X, y=None):
        self._validate_inputs(X)

        scaffold_to_indices = defaultdict(list)
        scaffolds = []

        for i, mol in enumerate(X):
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
            scaffolds.append(scaffold_smiles)
            scaffold_to_indices[scaffold_smiles].append(i)

        labels = [-1] * len(X)
        for cluster_id, indices in enumerate(scaffold_to_indices.values()):
            for idx in indices:
                labels[idx] = cluster_id

        self.labels_ = np.array(labels)
        self.clusters_ = list(scaffold_to_indices.values())
        self.scaffolds_ = list(scaffold_to_indices.keys())
        self.X_ = X
        return self

    def predict(self, X):
        self._validate_inputs(X)
        if self.scaffolds_ is None:
            raise ValueError("Model has not been fitted yet.")

        preds = []
        known_scaffolds = {smi: idx for idx, smi in enumerate(self.scaffolds_)}

        for mol in X:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
            preds.append(known_scaffolds.get(scaffold_smiles, -1))  # -1 = new/unknown scaffold

        return np.array(preds)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)
