from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

import pandas as pd
import matplotlib.pyplot as plt
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


def generate_tsne(
    csv_path,
    labels,
    n_bits=2048,
    perplexity=30.0,
    random_state=42,
    save_path="tsne_plot.png",
    metrics_info=None,
):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Smiles"])
    smiles = df["Smiles"].tolist()

    fingerprints = smiles_to_fingerprints(smiles, n_bits=n_bits)
    if len(fingerprints) == 0:
        raise ValueError("No valid fingerprints found.")

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(fingerprints)

    plt.figure(figsize=(10, 8))

    labels = np.array(labels)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            color=colors[i],
            label=f"Cluster {label}",
            alpha=0.7,
            s=20,
        )

    title = "t-SNE visualization of clustered molecules"
    if metrics_info:
        title += f"\n{metrics_info}"

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="upper right", title="Clusters")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")
