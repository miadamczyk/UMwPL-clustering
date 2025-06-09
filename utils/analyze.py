from collections import Counter
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


def analyze_structural_clusters(df, cluster_labels):
    # Generowanie szkieletów Murcko
    df['Scaffold'] = df['Smiles'].apply(lambda s: MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)))

    # Analiza dominujących szkieletów w klastrach
    cluster_scaffolds = {}
    for cluster_id in set(cluster_labels):
        cluster_smiles = df[cluster_labels == cluster_id]['Smiles']
        scaffolds = [MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)) for s in cluster_smiles]
        scaffold_counts = Counter(scaffolds)
        cluster_scaffolds[cluster_id] = scaffold_counts.most_common(3)  # Top 3 szkieletów

    # Wizualizacja
    for cluster, scaffolds in cluster_scaffolds.items():
        print(f"\nCluster {cluster} dominant scaffolds:")
        for scaff, count in scaffolds:
            print(f"  {Chem.MolToSmiles(scaff)}: {count} compounds")


def analyze_source_distribution(df, cluster_labels):
    source_cluster_table = pd.crosstab(
        df['Source Description'],
        cluster_labels,
        normalize='index'
    )

    # Wizualizacja
    plt.figure(figsize=(12, 8))
    sns.heatmap(source_cluster_table, annot=True, cmap="YlGnBu")
    plt.title("Distribution of Data Sources Across Clusters")
    plt.show()

    # Test istotności chi-kwadrat
    chi2, p, _, _ = chi2_contingency(pd.crosstab(df['Source Description'], cluster_labels))
    print(f"Chi-square test p-value: {p:.4f}")
    if p < 0.05:
        print("Significant correlation between clusters and data sources")




def analyze_assay_consistency(df, cluster_labels):
    # Dodaj etykiety klastrów jako nową kolumnę
    df = df.copy()
    df['Cluster'] = cluster_labels

    try:
        # Grupowanie po klastrach i testach
        assay_cluster_stats = df.groupby(['Assay Description', 'Cluster'])['Standard Value'].agg(
            ['mean', 'std', 'count']
        ).reset_index()

        # Obliczanie wariancji między testami w ramach klastra
        if not assay_cluster_stats.empty:
            cluster_variability = assay_cluster_stats.groupby('Cluster')['std'].mean()
            print("\nAverage standard deviation per cluster:")
            print(cluster_variability)

        # Wykres pudełkowy aktywności w klastrach
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y='Standard Value', data=df)
        plt.title("Activity Value Distribution per Cluster")
        plt.ylabel("Standard Value")
        plt.xlabel("Cluster")
        plt.show()

        # Analiza dla związków testowanych wielokrotnie
        duplicate_smiles = df[df.duplicated('Smiles', keep=False)]
        if not duplicate_smiles.empty:
            assay_agreement = duplicate_smiles.groupby('Smiles')['Standard Value'].std().reset_index(name='std_dev')
            high_variance = assay_agreement[assay_agreement['std_dev'] > 1.0]

            print("\nCompounds with high assay variance (>1.0):")
            print(high_variance)

            # Wykres dla związków z wieloma pomiarami
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='Smiles', y='Standard Value', data=duplicate_smiles)
            plt.title("Activity Variance for Compounds with Multiple Measurements")
            plt.xticks(rotation=90)
            plt.show()
        else:
            print("No compounds with multiple measurements found.")

    except KeyError as e:
        print(f"Missing column in data: {e}")
    except Exception as e:
        print(f"Error in assay consistency analysis: {e}")
