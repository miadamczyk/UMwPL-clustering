from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from rdkit import Chem
from utils.loader import FingerprintLoader

import numpy as np


class KMeansFingerprintClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, n_bits=2048, random_state=None):
        self.n_clusters = n_clusters
        self.n_bits = n_bits
        self.random_state = random_state
        self.kmeans_model = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.loader = FingerprintLoader(n_bits)

    def fit(self, X, y=None):
        features = self.loader.load_and_transform(X)
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state
        )
        self.kmeans_model.fit(features)
        self.labels_ = self.kmeans_model.labels_
        self.cluster_centers_ = self.kmeans_model.cluster_centers_
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        if isinstance(X, str):
            X_feat = self.loader.load_and_transform(X)
            return self.kmeans_model.predict(X_feat)
        else:
            if isinstance(X, list):
                fps = []
                for sm in X:
                    mol = Chem.MolFromSmiles(sm)
                    bitvect = np.zeros((self.n_bits,), dtype=int)
                    if mol is not None:
                        fp = self.loader.fp_generator.GetFingerprint(mol)
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
