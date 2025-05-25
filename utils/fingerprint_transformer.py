# transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
import numpy as np


class FingerprintTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fp_type='morgan', n_bits=2048, radius=2):
        self.fp_type = fp_type
        self.n_bits = n_bits
        self.radius = radius

        self.fp_generator = None

    def fit(self, X, y=None):
        if self.fp_type == 'morgan':
            self.fp_generator = AllChem.GetMorganGenerator(
                radius=self.radius, fpSize=self.n_bits
            )
        return self

    def _compute_fingerprint(self, mol):
        if mol is None:
            return None

        if self.fp_type == 'morgan':
            fp = self.fp_generator.GetFingerprint(mol)

        elif self.fp_type == 'maccs':
            fp = MACCSkeys.GenMACCSKeys(mol)

        elif self.fp_type == 'rdk':
            fp = RDKFingerprint(mol, fpSize=self.n_bits)

        else:
            raise ValueError(f"Unsupported fingerprint type: {self.fp_type}")

        return fp

    def _is_valid_smiles(self, sm):
        try:
            mol = Chem.MolFromSmiles(sm)
            return mol
        except Exception:
            return False

    def transform(self, X):
        X = np.ravel(X).tolist()
        fps = []
        for sm in X:
            mol = self._is_valid_smiles(sm)
            if not mol:
                continue
            fp = self._compute_fingerprint(mol)
            if fp is not None:
                fps.append(fp)
        return fps

    def get_rdkit_fingerprints(self, X):
        fps = []
        for sm in X:
            mol = Chem.MolFromSmiles(sm)
            fp = self._compute_fingerprint(mol)
            fps.append(fp)
        return fps
