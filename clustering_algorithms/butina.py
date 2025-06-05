from sklearn.base import BaseEstimator, ClusterMixin
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.SimDivFilters import rdSimDivPickers
import numpy as np


class ButinaFingerprintClustering(BaseEstimator, ClusterMixin):
    def __init__(self, similarity_threshold=0.85):
        self.similarity_threshold = similarity_threshold
        self.labels_ = None
        self.clusters_ = None
        self.leader_indices_ = None
        self.X_ = None

    def _validate_inputs(self, X):
        if not isinstance(X, (list, tuple)):
            raise TypeError("Input X must be a list or tuple of fingerprints.")
        for i, fp in enumerate(X):
            if fp is None:
                raise ValueError(f"Element at index {i} is None.")
            if not isinstance(fp, ExplicitBitVect):
                raise TypeError(f"Element {i} is not an RDKit ExplicitBitVect fingerprint.")

    def fit(self, X, y=None):
        self._validate_inputs(X)

        lp = rdSimDivPickers.LeaderPicker()
        leader_indices = lp.LazyBitVectorPick(X, len(X), 1.0 - self.similarity_threshold)

        labels = [-1] * len(X)
        clusters = [[] for _ in leader_indices]

        leaders = [X[i] for i in leader_indices]

        for i, fp in enumerate(X):
            sims = BulkTanimotoSimilarity(fp, leaders)
            best_cluster = int(np.argmax(sims))
            labels[i] = best_cluster
            clusters[best_cluster].append(i)

        self.labels_ = np.array(labels)
        self.clusters_ = clusters
        self.leader_indices_ = leader_indices
        self.X_ = X
        return self

    def predict(self, X):
        self._validate_inputs(X)
        if self.leader_indices_ is None or self.X_ is None:
            raise ValueError("Model has not been fitted yet.")

        leaders = [self.X_[i] for i in self.leader_indices_]
        preds = []
        for fp in X:
            sims = BulkTanimotoSimilarity(fp, leaders)
            best_cluster = int(np.argmax(sims))
            preds.append(best_cluster)
        return np.array(preds)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)
