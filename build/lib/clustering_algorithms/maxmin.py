from sklearn.base import BaseEstimator, ClusterMixin
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity, ExplicitBitVect
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
import numpy as np


class MaxMinFingerprintClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def _validate_inputs(self, X):
        if not isinstance(X, (list, tuple)):
            raise TypeError("Input X must be a list or tuple of fingerprints.")
        for i, fp in enumerate(X):
            if fp is not None and not isinstance(fp, ExplicitBitVect):
                raise TypeError(f"Element {i} is not an RDKit ExplicitBitVect fingerprint.")

    def fit(self, X, y=None):
        self._validate_inputs(X)

        fps_valid = [fp for fp in X if fp is not None]
        n_samples = len(fps_valid)

        if n_samples == 0:
            raise ValueError("No valid fingerprints provided.")
        if not (0 < self.n_clusters <= n_samples):
            raise ValueError(f"Invalid number of clusters: {self.n_clusters}")

        picker = MaxMinPicker()

        def dist_func(i, j):
            return 1.0 - TanimotoSimilarity(fps_valid[i], fps_valid[j])

        centers_idx = picker.LazyPick(dist_func, n_samples, self.n_clusters, seed=self.random_state)
        self.cluster_centers_ = [fps_valid[i] for i in centers_idx]

        return self

    def predict(self, X):
        self._validate_inputs(X)

        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet.")

        labels = []
        for fp in X:
            if fp is None:
                labels.append(-1)
                continue
            sims = [TanimotoSimilarity(fp, center) for center in self.cluster_centers_]
            labels.append(int(np.argmax(sims)))
        self.labels_ = np.array(labels)
        return self.labels_

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)
