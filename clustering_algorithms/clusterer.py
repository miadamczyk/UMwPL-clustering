from sklearn.cluster import Birch
from sklearn_extra.cluster import KMedoids
from clustering_algorithms.maxmin import MaxMinFingerprintClustering
from clustering_algorithms.butina import ButinaFingerprintClustering
from clustering_algorithms.scaffold import ScaffoldClustering
from clustering_algorithms.jarvis_patrick import JarvisPatrickClustering
from sklearn.pipeline import Pipeline
from utils.fingerprint_transformer import FingerprintTransformer


def get_clusterer(
        name: str,
        n_clusters: int = 3,
        random_state: int = 42,
        fp_type: str = "morgan",
        neighbor_k: int = 5,
        min_shared: int = 3,
):
    name = name.lower()

    if name == "scaffold":
        transformer = FingerprintTransformer(fp_type=fp_type, return_mols=True)
        clusterer = ScaffoldClustering()
        return Pipeline([
            ("mol_conversion", transformer),
            ("clusterer", clusterer),
        ])

    if name == "jarvis-patrick":
        transformer = FingerprintTransformer(fp_type=fp_type, return_as_rdkit=True)
        clusterer = JarvisPatrickClustering(k=neighbor_k, k_min=min_shared)
        return Pipeline([
            ("fingerprints", transformer),
            ("clusterer", clusterer),
        ])

    if name == "kmedoids":
        transformer = FingerprintTransformer(fp_type=fp_type, return_distance_matrix=True)
        clusterer = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=random_state, max_iter=500)

    elif name == "maxmin":
        transformer = FingerprintTransformer(fp_type=fp_type, return_as_rdkit=True)
        clusterer = MaxMinFingerprintClustering(n_clusters=n_clusters, random_state=random_state)

    elif name == "butina":
        transformer = FingerprintTransformer(fp_type=fp_type, return_as_rdkit=True)
        clusterer = ButinaFingerprintClustering(similarity_threshold=0.6)

    elif name == "birch":
        transformer = FingerprintTransformer(fp_type=fp_type, return_distance_matrix=False)
        clusterer = Birch(n_clusters=n_clusters)

    else:
        raise ValueError(f"Unsupported clustering algorithm: {name}")

    return Pipeline([
        ("fingerprints", transformer),
        ("clusterer", clusterer),
    ])
