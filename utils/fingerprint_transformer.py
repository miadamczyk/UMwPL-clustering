from sklearn.base import BaseEstimator, TransformerMixin
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from rdkit import DataStructs
import numpy as np


class FingerprintTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fp_type='morgan', n_bits=2048, radius=2, return_distance_matrix=False):
        self.fp_type = fp_type
        self.n_bits = n_bits
        self.radius = radius
        self.return_distance_matrix = return_distance_matrix

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
            return self.fp_generator.GetFingerprint(mol)
        elif self.fp_type == 'maccs':
            return MACCSkeys.GenMACCSKeys(mol)
        elif self.fp_type == 'rdk':
            return RDKFingerprint(mol, fpSize=self.n_bits)
        else:
            raise ValueError(f"Unsupported fingerprint type: {self.fp_type}")

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
                fps.append(None)
                continue
            fp = self._compute_fingerprint(mol)
            fps.append(fp)

        if self.return_distance_matrix:
            n = len(fps)
            dist_matrix = np.ones((n, n))
            for i in range(n):
                if fps[i] is None:
                    continue
                for j in range(i + 1, n):
                    if fps[j] is None:
                        continue
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    dist = 1.0 - sim
                    dist_matrix[i, j] = dist_matrix[j, i] = dist
            return dist_matrix

        else:
            # Only return valid fps (e.g., for maxmin)
            return [fp for fp in fps if fp is not None]

    def get_rdkit_fingerprints(self, X):
        return self.transform(X)
