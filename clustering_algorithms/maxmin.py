from sklearn.base import BaseEstimator, ClusterMixin
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.ML.Cluster import MaxMinPicker
import numpy as np

class MaxMinFingerprintClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, fps, y=None):
        fps_valid = [fp for fp in fps if fp is not None]
        n_samples = len(fps_valid)
        if n_samples == 0:
            raise ValueError("No valid fingerprints.")
        if not (0 < self.n_clusters <= n_samples):
            raise ValueError(f"Invalid number of clusters: {self.n_clusters}")

        picker = MaxMinPicker()

        def dist_func(i, j):
            return 1.0 - FingerprintSimilarity(fps_valid[i], fps_valid[j])

        centers_idx = picker.LazyPick(self.n_clusters, n_samples, dist_func, seed=self.random_state)
        self.cluster_centers_ = [fps_valid[i] for i in centers_idx]

        return self

    def predict(self, fps):
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet.")

        labels = []
        for fp in fps:
            if fp is None:
                labels.append(-1)
                continue
            sims = [FingerprintSimilarity(fp, center) for center in self.cluster_centers_]
            labels.append(int(np.argmax(sims)))
        self.labels_ = np.array(labels)
        return self.labels_

    def fit_predict(self, fps, y=None):
        self.fit(fps, y)
        return self.predict(fps)
