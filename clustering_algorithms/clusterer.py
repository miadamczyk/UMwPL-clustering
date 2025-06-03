from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, SpectralClustering
from sklearn_extra.cluster import KMedoids
from clustering_algorithms.maxmin import MaxMinFingerprintClustering
from clustering_algorithms.butina import ButinaFingerprintClustering
from sklearn.pipeline import Pipeline
from utils.fingerprint_transformer import FingerprintTransformer


def get_clusterer(name: str, n_clusters: int = 3, random_state: int = 42, fp_type: str = "morgan"):
    if name == "kmedoids":
        transformer = FingerprintTransformer(fp_type=fp_type, return_distance_matrix=True)
        clusterer = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=random_state)

    elif name in ["maxmin", "butina"]:
        transformer = FingerprintTransformer(fp_type=fp_type, return_as_rdkit=True)
        if name == "maxmin":
            clusterer = MaxMinFingerprintClustering(n_clusters=n_clusters, random_state=random_state)
        else:
            clusterer = ButinaFingerprintClustering(similarity_threshold=0.6)

    else:
        transformer = FingerprintTransformer(fp_type=fp_type, return_distance_matrix=False)
        clustering_algorithms = {
            "kmeans": KMeans(n_clusters=n_clusters, random_state=random_state),
            "agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
            "birch": Birch(n_clusters=n_clusters),
            "spectral": SpectralClustering(n_clusters=n_clusters),
        }

        if name not in clustering_algorithms:
            raise ValueError(f"Unsupported clustering algorithm: {name}")

        clusterer = clustering_algorithms[name]

    pipeline = Pipeline([
        ("fingerprints", transformer),
        ("clusterer", clusterer),
    ])

    return pipeline
