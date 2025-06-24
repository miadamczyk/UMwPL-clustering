# Molecular Structural Clustering

This project enables clustering of molecular data (SMILES) using various algorithms, either provided by scikit-learn or additionally implemented in a scikit-learn-like interface: `kmedoids`, `maxmin`, `butina`, `scaffold`, `jarvis-patrick`, `birch`. It also supports cluster quality evaluation through various metrics and generates 2D visualizations (`t-SNE`, `UMAP`).

---

# Input Data Format

To perform clustering on real datasets from ChEMBL, the file should contain the following columns: `Molecule ChEMBL ID`, `Standard Value`, and `Smiles`. Such data can be obtained, for example, from [ChEMBL](https://www.ebi.ac.uk/chembl/).

The dataset can be cleaned using the script:

```bash
python utils/clean_data.py --input raw_data.csv --output cleaned.csv
```

---

# Running the Project

After installing the project, you can run clustering:

```bash
python main.py --csv data.csv --algo butina --tsne
```

---

# Command-line Arguments

| Argument        | Description |
|-----------------|-------------|
| `--csv`, `-f`   | Path to the CSV file with a `Smiles` column (**required**) |
| `--algo`, `-a`  | Clustering algorithm: `kmedoids`, `birch`, `maxmin`, `butina`, `scaffold`, `jarvis-patrick` |
| `--clusters`, `-k` | Number of clusters (if required by the algorithm) |
| `--fp_type`     | Type of molecular fingerprint (e.g., `morgan`) |
| `--neighbor_k`  | Number of neighbors (for `jarvis-patrick`) |
| `--min_shared`  | Minimum number of shared neighbors (`jarvis-patrick`) |
| `--tsne`        | Generates a t-SNE plot |
| `--umap`        | Generates a UMAP plot |
| `--metric`, `-m`| Evaluation metrics: `silhouette_score`, `adjusted_rand_score`, `mutual_info_score`, `normalized_mutual_info_score`, `homogeneity_score`, `completeness_score`, `v_measure_score` |

---

# Output

- Clustering evaluation metrics are printed to the console.
- `.png` visualizations (t-SNE and/or UMAP) are saved to the working directory.

---

# Environment Setup

1. Make sure Python **3.8** is installed.
2. (Optional) Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. Install the project:

```bash
pip install .
```

---

# Dependencies

Defined in `pyproject.toml`, including:

- `numpy`
- `pandas`
- `rdkit`
- `scikit-learn`
- `scikit-learn-extra`
- `umap-learn`
- `matplotlib` (>=3.7,<3.8)
- `ruff` (for linting and formatting)

---

# Code Style and Linting with `ruff`

This project uses [`ruff`](https://docs.astral.sh/ruff/) for fast Python linting and automatic formatting.

# Check for issues:

```bash
ruff check .
```

# Auto-fix issues:

```bash
ruff check . --fix
```

# Install `ruff`:

```bash
pip install ruff
```

---

# Project Structure

```
main.py                  # Main script
--- clustering_algorithms/   # Clustering algorithm implementations
--- utils/                   # Visualization, metrics, data preprocessing
--- pyproject.toml           # Project configuration and dependencies
--- README.md                # This file
```

---

# Authors

- Miłosz Adamczyk  
- Illia Dovhalenko  
- Kacper Marzol  
- Filip Skałka  

---

# License

Released under the **MIT License**. See the `LICENSE` file for details.