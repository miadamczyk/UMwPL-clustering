#!/usr/bin/env python3
from clustering_algorithms.kmeans import KMeansFingerprintClustering
from clustering_algorithms.maxmin import MaxMinFingerprintClustering
from utils.create_tsne import generate_tsne
from utils.create_umap import generate_umap

import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run clustering on molecular data using specified algorithm."
    )
    parser.add_argument(
        "--clusters", "-k", type=int, required=True,
        help="Number of clusters to generate."
    )
    parser.add_argument(
        "--csv", "-f", type=str, required=True,
        help="Path to the CSV file with 'Smiles' column."
    )
    parser.add_argument(
        "--algo", "-a", type=str, choices=["kmeans", "maxmin"], default="kmeans",
        help="Clustering algorithm to use (default: kmeans)."
    )
    parser.add_argument(
        "--tsne", action="store_true",
        help="If set, generates a t-SNE plot after clustering."
    )
    parser.add_argument(
        "--umap", action="store_true",
        help="If set, generates a UMAP plot after clustering."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.algo == "kmeans":
        clusterer = KMeansFingerprintClustering(n_clusters=args.clusters, random_state=42)
    elif args.algo == "maxmin":
        clusterer = MaxMinFingerprintClustering(n_clusters=args.clusters, random_state=42)

    print(f"Running {args.algo.upper()} clustering on: {args.csv}")
    labels = clusterer.fit_predict(args.csv)

    print("\n=== Cluster Assignments ===")
    for idx, label in enumerate(labels):
        print(f"Sample {idx + 1}: Cluster {label}")

    csv_name = os.path.splitext(os.path.basename(args.csv))[0]
    num_clusters = int(args.clusters)

    if args.tsne:
        print("\nGenerating t-SNE visualization...")
        save_path = f"tsne_{csv_name}-{args.algo}-num_clusters_{num_clusters}.png"
        generate_tsne(args.csv, labels, save_path=save_path)

    if args.umap:
        print("\nGenerating UMAP visualization...")
        save_path = f"umap_{csv_name}-{args.algo}-num_clusters_{num_clusters}.png"
        generate_umap(args.csv, labels, save_path=save_path)

if __name__ == "__main__":
    main()
