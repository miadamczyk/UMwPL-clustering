from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

import pandas as pd
import matplotlib.pyplot as plt
import umap
import numpy as np


def smiles_to_fingerprints(smiles_list, n_bits=2048):
    fps = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=int)
            ConvertToNumpyArray(fp, arr)
            fps.append(arr)
    return np.array(fps)


def generate_umap(
    csv_path,
    labels,
    n_bits=2048,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    save_path="umap_plot.png",
    metrics_info=None,
):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Smiles"])
    smiles = df["Smiles"].tolist()

    fingerprints = smiles_to_fingerprints(smiles, n_bits=n_bits)
    if len(fingerprints) == 0:
        raise ValueError("No valid fingerprints found.")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    umap_results = reducer.fit_transform(fingerprints)

    plt.figure(figsize=(10, 8))

    labels = np.array(labels)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(
            umap_results[indices, 0],
            umap_results[indices, 1],
            color=colors[i],
            label=f"Cluster {label}",
            alpha=0.7,
            s=20,
        )

    title = "UMAP visualization of clustered molecules"
    if metrics_info:
        title += f"\n{metrics_info}"

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(loc="upper right", title="Clusters")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"UMAP plot saved to {save_path}")
