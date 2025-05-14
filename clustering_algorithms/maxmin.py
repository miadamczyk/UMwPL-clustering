from sklearn.base import BaseEstimator, ClusterMixin
from rdkit.DataStructs import FingerprintSimilarity
from utils.loader import (
    FingerprintLoader,
)  # assuming you save it as fingerprint_loader.py

import numpy as np


class MaxMinFingerprintClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, n_bits=2048, random_state=None):
        self.n_clusters = n_clusters
        self.n_bits = n_bits
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None
        self.loader = FingerprintLoader(n_bits)

    def fit(self, X, y=None):
        fps = self.loader.load_fingerprints(X)
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
                sim = FingerprintSimilarity(fps_valid[centers_idx[-1]], fps_valid[idx])
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

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_
