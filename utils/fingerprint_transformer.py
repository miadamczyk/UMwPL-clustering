from sklearn.base import BaseEstimator, TransformerMixin
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys, RDKFingerprint, rdFingerprintGenerator
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

def smiles_to_fingerprints(smiles_list, n_bits=2048, radius=2):
    fps = []
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((n_bits,), dtype=int)
            ConvertToNumpyArray(fp, arr)
            fps.append(arr)
    return np.array(fps)

class FingerprintTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fp_type='morgan', n_bits=2048, radius=2,
                 return_distance_matrix=False, return_as_rdkit=False,
                 return_mols=False):  # New flag here
        self.fp_type = fp_type
        self.n_bits = n_bits
        self.radius = radius
        self.return_distance_matrix = return_distance_matrix
        self.return_as_rdkit = return_as_rdkit
        self.return_mols = return_mols
        self.fp_generator = None

    def fit(self, X, y=None):
        if self.fp_type == 'morgan':
            self.fp_generator = GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
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

    def _smiles_to_mol(self, sm):
        try:
            return Chem.MolFromSmiles(sm)
        except Exception:
            return None

    def transform(self, X):
        X = np.ravel(X).tolist()
        if self.return_mols:
            mols = []
            for sm in X:
                mol = self._smiles_to_mol(sm)
                mols.append(mol)
            return mols

        fps = []
        for sm in X:
            mol = self._smiles_to_mol(sm)
            if mol is None:
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

        elif self.return_as_rdkit:
            return fps

        else:
            fp_arrays = []
            for fp in fps:
                if fp is None:
                    continue
                arr = np.zeros((self.n_bits,), dtype=int)
                ConvertToNumpyArray(fp, arr)
                fp_arrays.append(arr)
            return np.array(fp_arrays)

    def get_rdkit_fingerprints(self, X):
        return self.transform(X)
