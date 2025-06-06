from sklearn.base import BaseEstimator, ClusterMixin
from rdkit.DataStructs import BulkTanimotoSimilarity
import numpy as np
from collections import defaultdict

class JarvisPatrickClustering(BaseEstimator, ClusterMixin):
    def __init__(self, k=5, k_min=3):
        self.k = k
        self.k_min = k_min
        self.labels_ = None
        self.X_ = None
        self.neighbor_lists_ = None

    def _validate_inputs(self, X):
        from rdkit.DataStructs.cDataStructs import ExplicitBitVect
        if not isinstance(X, (list, tuple)):
            raise TypeError("Input must be a list or tuple of RDKit fingerprints.")
        for i, fp in enumerate(X):
            if fp is None or not isinstance(fp, ExplicitBitVect):
                raise ValueError(f"Element {i} is not a valid RDKit ExplicitBitVect fingerprint.")

    def _compute_neighbors(self, X):
        n = len(X)
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            sims = BulkTanimotoSimilarity(X[i], X)
            sims[i] = -1.0  # exclude self similarity
            top_k = np.argsort(sims)[-self.k:]
            neighbors[i] = set(top_k)
        return neighbors

    def fit(self, X, y=None):
        self._validate_inputs(X)
        self.X_ = X
        neighbor_lists = self._compute_neighbors(X)
        self.neighbor_lists_ = neighbor_lists

        n = len(X)
        labels = [-1] * n
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue
            labels[i] = cluster_id
            for j in range(i + 1, n):
                if labels[j] != -1:
                    continue
                if (
                    i in neighbor_lists[j]
                    and j in neighbor_lists[i]
                    and len(neighbor_lists[i] & neighbor_lists[j]) >= self.k_min
                ):
                    labels[j] = cluster_id
            cluster_id += 1

        self.labels_ = np.array(labels)
        return self

    def predict(self, X):
        self._validate_inputs(X)
        if self.X_ is None or self.labels_ is None:
            raise ValueError("The model must be fitted before prediction.")

        predictions = []
        known_clusters = defaultdict(list)
        for idx, label in enumerate(self.labels_):
            if label != -1:
                known_clusters[label].append(idx)

        for x in X:
            sims = BulkTanimotoSimilarity(x, self.X_)
            top_k_indices = np.argsort(sims)[-self.k:]
            top_k_set = set(top_k_indices)

            best_cluster = -1
            for cluster_id, member_indices in known_clusters.items():
                mutual_neighbors = [i for i in member_indices if i in top_k_set]
                reverse_neighbors = 0

                for i in mutual_neighbors:
                    if len(self.neighbor_lists_[i].intersection(top_k_set)) >= self.k_min:
                        reverse_neighbors += 1

                if reverse_neighbors >= self.k_min:
                    best_cluster = cluster_id
                    break

            predictions.append(best_cluster)

        return np.array(predictions)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
