#!/usr/bin/env python3
from clustering_algorithms.kmeans import KMeansFingerprintClustering
from clustering_algorithms.maxmin import MaxMinFingerprintClustering
from utils.create_tsne import generate_tsne
from utils.create_umap import generate_umap
from utils.metrics import evaluate_clustering

import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run clustering on molecular data using specified algorithm."
    )
    parser.add_argument(
        "--clusters",
        "-k",
        type=int,
        required=True,
        help="Number of clusters to generate.",
    )
    parser.add_argument(
        "--csv",
        "-f",
        type=str,
        required=True,
        help="Path to the CSV file with 'Smiles' column.",
    )
    parser.add_argument(
        "--algo",
        "-a",
        type=str,
        choices=["kmeans", "maxmin"],
        default="kmeans",
        help="Clustering algorithm to use (default: kmeans).",
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="If set, generates a t-SNE plot after clustering.",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        help="If set, generates a UMAP plot after clustering.",
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        choices=[
            "silhouette_score",
            "adjusted_rand_score",
            "mutual_info_score",
            "normalized_mutual_info_score",
            "homogeneity_score",
            "completeness_score",
            "v_measure_score",
        ],
        nargs="+",
        help="Optional: clustering evaluation metric(s) to compute. You can specify multiple metrics.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.algo == "kmeans":
        clusterer = KMeansFingerprintClustering(
            n_clusters=args.clusters, random_state=42
        )
    elif args.algo == "maxmin":
        clusterer = MaxMinFingerprintClustering(
            n_clusters=args.clusters, random_state=42
        )

    print(f"Running {args.algo.upper()} clustering on: {args.csv}")
    labels = clusterer.fit_predict(args.csv)

    print("\n=== Cluster Assignments ===")
    for idx, label in enumerate(labels):
        print(f"Sample {idx + 1}: Cluster {label}")

    metric_scores = {}
    if args.metric:
        for metric in args.metric:
            print(f"\nEvaluating clustering with metric: {metric}")
            try:
                score = evaluate_clustering(args.csv, labels, metric)
                metric_scores[metric] = score
                print(f"{metric}: {score:.4f}")
            except Exception as e:
                print(f"Error computing {metric}: {e}")

    metrics_info = "\n".join(
        [f"{metric}: {score:.4f}" for metric, score in metric_scores.items()]
    )

    csv_name = os.path.splitext(os.path.basename(args.csv))[0]
    num_clusters = int(args.clusters)

    if args.tsne or args.umap:
        print("\nGenerating visualization...")

        # Pass the metrics information to the visualization function
        if args.tsne:
            save_path = f"tsne_{csv_name}-{args.algo}-clusters_{num_clusters}.png"
            generate_tsne(
                args.csv, labels, save_path=save_path, metrics_info=metrics_info
            )

        if args.umap:
            save_path = f"umap_{csv_name}-{args.algo}-clusters_{num_clusters}.png"
            generate_umap(
                args.csv, labels, save_path=save_path, metrics_info=metrics_info
            )


if __name__ == "__main__":
    main()
