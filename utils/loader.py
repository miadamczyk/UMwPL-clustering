import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganGenerator


class FingerprintLoader:
    def __init__(self, n_bits=2048):
        self.n_bits = n_bits
        self.fp_generator = GetMorganGenerator(radius=2, fpSize=n_bits)

    def load_fingerprints(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["Smiles"])
        smiles_list = df["Smiles"].tolist()
        fps = []
        for sm in smiles_list:
            mol = Chem.MolFromSmiles(sm)
            fp = self.fp_generator.GetFingerprint(mol) if mol is not None else None
            fps.append(fp)
        return fps

    def load_and_transform(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["Smiles", "Standard Value"])
        smiles_list = df["Smiles"].tolist()
        values = df["Standard Value"].astype(float).to_numpy().reshape(-1, 1)

        fps = []
        for sm in smiles_list:
            mol = Chem.MolFromSmiles(sm)
            bitvect = np.zeros((self.n_bits,), dtype=int)
            if mol is not None:
                fp = self.fp_generator.GetFingerprint(mol)
                for i in range(self.n_bits):
                    if fp.GetBit(i):
                        bitvect[i] = 1
            fps.append(bitvect)
        fps = np.array(fps)
        X = np.hstack([fps, values])
        return X
