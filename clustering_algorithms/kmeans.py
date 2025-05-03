from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganGenerator

import numpy as np
import pandas as pd


class KMeansFingerprintClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, n_bits=2048, random_state=None):
        self.n_clusters = n_clusters
        self.n_bits = n_bits
        self.random_state = random_state
        self.kmeans_model = None
        self.labels_ = None
        self.cluster_centers_ = None

        self.fp_generator = GetMorganGenerator(radius=2, fpSize=self.n_bits)

    def _load_and_transform(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Smiles', 'Standard Value'])
        smiles_list = df['Smiles'].tolist()
        values = df['Standard Value'].astype(float).to_numpy().reshape(-1, 1)

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

    def fit(self, csv_path):
        X = self._load_and_transform(csv_path)
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans_model.fit(X)
        self.labels_ = self.kmeans_model.labels_
        self.cluster_centers_ = self.kmeans_model.cluster_centers_
        return self

    def predict(self, X):
        if isinstance(X, str):
            X_feat = self._load_and_transform(X)
            return self.kmeans_model.predict(X_feat)
        else:
            if isinstance(X, list):
                fps = []
                for sm in X:
                    mol = Chem.MolFromSmiles(sm)
                    bitvect = np.zeros((self.n_bits,), dtype=int)
                    if mol is not None:
                        fp = self.fp_generator.GetFingerprint(mol)
                        for i in range(self.n_bits):
                            if fp.GetBit(i):
                                bitvect[i] = 1
                    fps.append(bitvect)
                fps = np.array(fps)
                values = np.zeros((fps.shape[0], 1))
                X_feat = np.hstack([fps, values])
                return self.kmeans_model.predict(X_feat)
            else:
                return self.kmeans_model.predict(X)

    def fit_predict(self, csv_path):
        self.fit(csv_path)
        return self.labels_
