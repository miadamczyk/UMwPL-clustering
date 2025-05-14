from sklearn.metrics import (
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
)
from utils.loader import FingerprintLoader

import pandas as pd


def compute_metric(metric_name, true_vals, features, labels):
    if metric_name == "adjusted_rand_score":
        return adjusted_rand_score(true_vals, labels)
    elif metric_name == "mutual_info_score":
        return mutual_info_score(true_vals, labels)
    elif metric_name == "normalized_mutual_info_score":
        return normalized_mutual_info_score(true_vals, labels)
    elif metric_name == "homogeneity_score":
        return homogeneity_score(true_vals, labels)
    elif metric_name == "completeness_score":
        return completeness_score(true_vals, labels)
    elif metric_name == "v_measure_score":
        return v_measure_score(true_vals, labels)
    elif metric_name == "silhouette_score":
        return silhouette_score(features, labels)
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")


def evaluate_clustering(csv_path, labels, metric_name, n_bits=2048):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Smiles", "Standard Value"])

    true_vals = df["Standard Value"].astype(float).values
    loader = FingerprintLoader(n_bits=n_bits)
    features = loader.load_and_transform(csv_path)

    if metric_name == "silhouette_score":
        return compute_metric(metric_name, None, features, labels)
    else:
        return compute_metric(metric_name, true_vals, features, labels)
