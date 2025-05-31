# clusterer_factory.py

from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, SpectralClustering
from clustering_algorithms.maxmin import MaxMinFingerprintClustering
from sklearn.pipeline import Pipeline
from utils.fingerprint_transformer import FingerprintTransformer


def get_clusterer(name: str, n_clusters: int = 3, random_state: int = 42, fp_type: str = "morgan"):
    clustering_algorithms = {
        "kmeans": KMeans(n_clusters=n_clusters, random_state=random_state),
        "maxmin": MaxMinFingerprintClustering(n_clusters=n_clusters, random_state=random_state),
        "agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
        "birch": Birch(n_clusters=n_clusters),
        "spectral": SpectralClustering(n_clusters=n_clusters),
    }

    if name not in clustering_algorithms:
        raise ValueError(f"Unsupported clustering algorithm: {name}")

    pipeline = Pipeline([
        ("fingerprints", FingerprintTransformer(fp_type=fp_type)),
        ("clusterer", clustering_algorithms[name]),
    ])

    return pipeline
