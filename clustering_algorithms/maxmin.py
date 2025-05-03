from sklearn.base import BaseEstimator, ClusterMixin
from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem.AllChem import GetMorganGenerator

import numpy as np
import pandas as pd


class MaxMinFingerprintClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, n_bits=2048, random_state=None):
        self.n_clusters = n_clusters
        self.n_bits = n_bits
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None

        self.fp_generator = GetMorganGenerator(radius=2, fpSize=self.n_bits)

    def _load_fingerprints(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Smiles'])
        smiles_list = df['Smiles'].tolist()
        fps = []
        for sm in smiles_list:
            mol = Chem.MolFromSmiles(sm)
            fp = self.fp_generator.GetFingerprint(mol) if mol is not None else None
            fps.append(fp)
        return fps

    def fit(self, csv_path):
        fps = self._load_fingerprints(csv_path)
        fps_valid = [fp for fp in fps if fp is not None]
        n_samples = len(fps_valid)
        if n_samples == 0:
            raise ValueError("No data.")
        if not (0 < self.n_clusters <= n_samples):
            raise ValueError(f"Num of clusters {self.n_clusters} is wrong.")

        rng = np.random.default_rng(self.random_state)
        first_idx = int(rng.integers(0, n_samples))
        centers_idx = [first_idx]
        min_dists = np.full(n_samples, np.inf)
        min_dists[first_idx] = 0.0

        for _ in range(1, self.n_clusters):
            for idx in range(n_samples):
                if idx in centers_idx:
                    continue
                sim = FingerprintSimilarity(fps_valid[centers_idx[-1]], fps_valid[idx])  # Tanimoto
                dist = 1.0 - sim
                if dist < min_dists[idx]:
                    min_dists[idx] = dist
            next_idx = int(np.argmax(min_dists))
            centers_idx.append(next_idx)
            min_dists[next_idx] = 0.0

        centers = [fps_valid[i] for i in centers_idx]
        self.cluster_centers_ = centers

        labels = []
        for fp in fps:
            if fp is None:
                labels.append(-1)
            else:
                sims = [FingerprintSimilarity(fp, center) for center in centers]
                labels.append(int(np.argmax(sims)))
        self.labels_ = np.array(labels)
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted (cluster_centers == None).")
        if isinstance(X, str):
            fps_new = self._load_fingerprints(X)
        else:
            fps_new = []
            for sm in X:
                mol = Chem.MolFromSmiles(sm)
                fp = self.fp_generator.GetFingerprint(mol) if mol is not None else None
                fps_new.append(fp)

        labels = []
        for fp in fps_new:
            if fp is None:
                labels.append(-1)
            else:
                sims = [FingerprintSimilarity(fp, center) for center in self.cluster_centers_]
                labels.append(int(np.argmax(sims)))
        return np.array(labels)

    def fit_predict(self, csv_path):
        self.fit(csv_path)
        return self.labels_
