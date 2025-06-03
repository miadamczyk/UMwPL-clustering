from sklearn_extra.cluster import KMedoids
from clustering_algorithms.maxmin import MaxMinFingerprintClustering
from sklearn.pipeline import Pipeline
from utils.fingerprint_transformer import FingerprintTransformer


def get_clusterer(name: str, n_clusters: int = 3, random_state: int = 42, fp_type: str = "morgan"):
    if name == "kmedoids":
        transformer = FingerprintTransformer(fp_type=fp_type, return_distance_matrix=True)
        clusterer = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=random_state)
    elif name == "maxmin":
        transformer = FingerprintTransformer(fp_type=fp_type, return_distance_matrix=False)
        clusterer = MaxMinFingerprintClustering(n_clusters=n_clusters, random_state=random_state)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {name}")

    pipeline = Pipeline([
        ("fingerprints", transformer),
        ("clusterer", clusterer),
    ])

    return pipeline
