#!/usr/bin/env python3

from clustering_algorithms.clusterer import get_clusterer
from utils.create_tsne import generate_tsne
from utils.create_umap import generate_umap
from utils.metrics import evaluate_clustering

import argparse
import os
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run clustering on molecular data using specified algorithm."
    )
    parser.add_argument(
        "--clusters", "-k", type=int, help="Number of clusters to generate (if applicable)."
    )
    parser.add_argument(
        "--csv", "-f", type=str, required=True, help="Path to the CSV file with 'Smiles' column."
    )
    parser.add_argument(
        "--algo", "-a",
        type=str,
        choices=[
            "kmedoids", "birch", "maxmin", "butina", "scaffold", "jarvis-patrick"
        ],
        default="butina",
        help="Clustering algorithm to use."
    )
    parser.add_argument(
        "--fp_type", type=str, default="morgan",
        help="Fingerprint type to use for clustering (if needed)."
    )
    parser.add_argument(
        "--neighbor_k", type=int, default=5,
        help="Number of nearest neighbors (Jarvis-Patrick only)."
    )
    parser.add_argument(
        "--min_shared", type=int, default=3,
        help="Minimum shared neighbors (Jarvis-Patrick only)."
    )
    parser.add_argument(
        "--tsne", action="store_true", help="Generate a t-SNE plot after clustering."
    )
    parser.add_argument(
        "--umap", action="store_true", help="Generate a UMAP plot after clustering."
    )
    parser.add_argument(
        "--metric", "-m",
        type=str,
        nargs="+",
        choices=[
            "silhouette_score", "adjusted_rand_score", "mutual_info_score",
            "normalized_mutual_info_score", "homogeneity_score",
            "completeness_score", "v_measure_score"
        ],
        help="Clustering evaluation metrics to compute (can specify multiple)."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    df = pd.read_csv(args.csv)
    smiles = df["Smiles"].dropna().tolist()

    print(f"Running {args.algo.upper()} clustering on: {args.csv}")

    try:
        cluster_kwargs = dict(
            n_clusters=args.clusters,
            random_state=42,
            fp_type=args.fp_type,
            neighbor_k=args.neighbor_k,
            min_shared=args.min_shared,
        )
        if args.algo in ["scaffold", "butina"]:
            cluster_kwargs.pop("n_clusters", None)

        pipeline = get_clusterer(args.algo, **cluster_kwargs)

    except ValueError as e:
        print(f"Error initializing clusterer: {e}")
        return

    try:
        labels = pipeline.fit_predict(smiles)
    except Exception as e:
        print(f"Clustering failed: {e}")
        return

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

    # Optional plots
    csv_name = os.path.splitext(os.path.basename(args.csv))[0]

    if args.tsne or args.umap:
        print("\nGenerating visualization...")

        if args.tsne:
            save_path = f"tsne_{csv_name}-{args.algo}-clusters.png"
            generate_tsne(args.csv, labels, save_path=save_path, metrics_info=metrics_info)

        if args.umap:
            save_path = f"umap_{csv_name}-{args.algo}-clusters.png"
            generate_umap(args.csv, labels, save_path=save_path, metrics_info=metrics_info)

if __name__ == "__main__":
    main()
