from sklearn.base import BaseEstimator, ClusterMixin
from rdkit.DataStructs.cDataStructs import ExplicitBitVect, BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina
import numpy as np


class ButinaFingerprintClustering(BaseEstimator, ClusterMixin):
    def __init__(self, similarity_threshold=0.85):
        self.valid_indices_ = None
        self.similarity_threshold = similarity_threshold
        self.labels_ = None
        self.clusters_ = None

    def _validate_inputs(self, X):
        if not isinstance(X, (list, tuple)):
            raise TypeError("Input X must be a list or tuple of fingerprints.")
        for i, fp in enumerate(X):
            if fp is not None and not isinstance(fp, ExplicitBitVect):
                raise TypeError(f"Element {i} is not an RDKit ExplicitBitVect fingerprint.")

    def fit(self, X, y=None):
        self._validate_inputs(X)
        fps_valid = [fp for fp in X if fp is not None]
        self.valid_indices_ = [i for i, fp in enumerate(X) if fp is not None]

        if len(fps_valid) == 0:
            raise ValueError("No valid fingerprints provided.")

        dists = []
        nfps = len(fps_valid)
        for i in range(1, nfps):
            sims = BulkTanimotoSimilarity(fps_valid[i], fps_valid[:i])
            dists.extend([1.0 - sim for sim in sims])

        clusters = Butina.ClusterData(dists, nfps, self.similarity_threshold, isDistData=True)
        self.clusters_ = clusters

        labels = [-1] * len(X)
        for cluster_id, cluster in enumerate(clusters):
            for local_idx in cluster:
                global_idx = self.valid_indices_[local_idx]
                labels[global_idx] = cluster_id

        self.labels_ = np.array(labels)
        return self

    def predict(self, X):
        raise NotImplementedError("Butina clustering does not support predicting new data.")

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_
