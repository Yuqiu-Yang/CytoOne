![Logo](/assets/logo.png)

# CytoOne

> A unified probabilistic framework for CyTOF data

[![Check build](https://github.com/Yuqiu-Yang/CytoOne/actions/workflows/push.yml/badge.svg)](https://github.com/Yuqiu-Yang/CytoOne/actions/workflows/push.yml)
[![Documentation Status](https://readthedocs.org/projects/cytoone/badge/?version=latest)](https://cytoone.readthedocs.io/en/latest/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](./LICENSE)
[![Python 3.9 | 3.10](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)](https://www.python.org)

![Model Overview](/assets/model_overview.png)

CytoOne is a single, coherent Bayesian deep generative model for mass cytometry
(CyTOF) data. Instead of stitching together task-specific tools — each with its
own, sometimes conflicting, distributional assumptions — CytoOne performs
**batch effect correction**, **differential expression analysis**, and
**visualization / dimension reduction** within one model trained end-to-end.

It combines a hierarchical latent architecture inspired by the Nouveau
Variational Autoencoder (NVAE) with a purpose-built **quasi zero-inflated
softplus-normal (QZIPN)** likelihood that matches the sparse, noisy
characteristics of CyTOF measurements.

📖 **Full documentation and step-by-step tutorials:**
**<https://cytoone.readthedocs.io>**

---

## Table of contents

- [Key features](#key-features)
- [Installation](#installation)
  - [Option 1: conda + pip](#option-1-conda--pip)
  - [Option 2: build from source](#option-2-build-from-source)
  - [Option 3: Docker / Apptainer (recommended for shared servers)](#option-3-docker--apptainer-recommended-for-shared-servers)
  - [Dependencies](#dependencies)
- [Input format](#input-format)
- [Example dataset](#example-dataset)
- [Quick start](#quick-start)
- [Tutorial](#tutorial)
  - [Interactive Python](#interactive-python)
  - [Command-line interface (CLI)](#command-line-interface-cli)
- [Outputs at a glance](#outputs-at-a-glance)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Key features

- **One model, three tasks.** Visualization, batch correction, and differential
  discovery share the same learned representation, so results are mutually
  consistent rather than the product of a heuristic pipeline.
- **A likelihood built for CyTOF.** The QZIPN likelihood captures
  zero-inflation, non-negativity, and the Gaussian noise added during Helios
  preprocessing.
- **Hierarchical latent space.** Global biology (e.g. cell types) is pushed to a
  2-D top layer for visualization, while lower layers retain the detail needed
  for faithful reconstruction and correction.
- **Scales to millions of cells** via stochastic variational inference.
- **Interactive Python *and* a CLI**, plus a containerized build for
  reproducible, install-free deployment.

---

## Installation

So far CytoOne has been tested on **Python 3.9 and 3.10**.

### Option 1: conda + pip

Create and activate an environment:

```shell
conda create -n cytoone python=3.10
conda activate cytoone
```

Install the stable release from PyPI:

```shell
pip install CytoOne
```

### Option 2: build from source

The latest features live on GitHub. Clone the repository, then build and
install locally:

```shell
git clone https://github.com/Yuqiu-Yang/CytoOne.git
cd CytoOne
python setup.py sdist bdist_wheel
```

A `dist/` folder will appear containing the wheel. Install it (replace
`VERSION` with the value you see in `dist/`):

```shell
cd ./dist
pip install ./CytoOne-VERSION-py3-none-any.whl
```

Confirm the installation:

```shell
python -m CytoOne --version
```

### Option 3: Docker / Apptainer (recommended for shared servers)

If your institution restricts direct installation on common server
infrastructure, use the container instead — nothing is installed on the host.

Pull the pre-built image:

```shell
docker pull ghcr.io/yuqiu-yang/cytoone:latest
docker run --rm ghcr.io/yuqiu-yang/cytoone:latest --version
```

…or build it yourself from the included [`Dockerfile`](./Dockerfile):

```shell
docker build -t cytoone:latest .
```

On HPC systems with Apptainer/Singularity instead of Docker:

```shell
apptainer pull cytoone.sif docker://ghcr.io/yuqiu-yang/cytoone:latest
apptainer run cytoone.sif --help
```

Full container usage — mounting data, GPU builds, JupyterLab, and
Apptainer/Singularity details — is documented in
[`docker/README.md`](./docker/README.md).

### Dependencies

With every install strategy the dependencies are resolved automatically. They
are listed here for reference:

- `python>=3.9,<3.11`
- `numpy<2.0`
- `pandas>=2.2.0`
- `anndata>=0.10,<0.11`
- `scanpy<1.11`
- `torch<2.0`
- `pyro-ppl<1.8.5`
- `seaborn`
- `jupyter`
- `ipywidgets`

---

## Input format

CytoOne takes up to two inputs:

1. **`cell_by_gene`** — a cell-by-marker matrix (a CSV path or a
   `pandas.DataFrame`).
   - Column names are protein markers; the **first column is the cell ID**.
   - CytoOne models the `arcsinh`-transformed protein measurements (cofactor
     5). Based on our experience the transformed range is roughly 0–10. If you
     see raw values in the hundreds, leave `normalize=True` so CytoOne applies
     the transform for you; if your data are already transformed, set
     `normalize=False`.

2. **`cell_metadata`** *(optional)* — per-cell annotations (a CSV path or a
   `pandas.DataFrame`).
   - **`cell_id`** — CytoOne assumes the **first column** holds the cell IDs;
     they must match those in `cell_by_gene`.
   - **`batch`** — batch annotation for each cell.
   - **`cell_type`** *(optional)* — used only for plotting, never by the model.
   - If `cell_metadata` is omitted, CytoOne assumes a single batch.
   - Your columns need not use these exact names — tell CytoOne which columns to
     use via `batch_index_col` and `celltype_col`.

A complete description of every column, the expected shapes, and common pitfalls
is in the [Input format guide](https://cytoone.readthedocs.io/en/latest/input_format.html).

---

## Example dataset

A small, ready-to-run dataset simulated with
[Cytomulate](https://github.com/kevin931/cytomulate) ships with the package
under [`tests/`](./tests) so you can try CytoOne without downloading anything:

| File | Shape | Description |
|------|-------|-------------|
| `tests/test_data_zi.csv` | 4000 × 10 | Zero-inflated marker measurements (raw-style) |
| `tests/test_data_n.csv` | 4000 × 10 | Noise-added marker measurements (Helios-style) |
| `tests/test_data_meta.csv` | 4000 × 2 | `batch` (2 batches) and `cell_type` (2 types) |

> The larger datasets used in the paper are archived on
> [Zenodo](https://zenodo.org/records/17795487).

---

## Quick start

Train CytoOne on the bundled example data and obtain an embedding in a few
lines:

```python
from CytoOne.cytoone_class import cytoone

cyto = cytoone(
    batch_index_col="batch",
    celltype_col="cell_type",
    normalize=True,        # apply arcsinh(./5)
    zero_inflated=True,    # the *_zi data are zero-inflated
    dr=True,               # also compute a reference UMAP
)

cyto.import_data(
    cell_by_gene="./tests/test_data_zi.csv",
    cell_metadata="./tests/test_data_meta.csv",
)

cyto.initialize_parameters()
cyto.training_loop(n_epoches=50)

# Two-dimensional, batch-effect-free embedding
_, z_samples = cyto.infer()
print(z_samples.head())
```

The equivalent one-liner with the CLI:

```shell
python -m CytoOne \
    --cell_by_gene ./tests/test_data_zi.csv \
    --cell_metadata ./tests/test_data_meta.csv \
    --batch_index_col batch --celltype_col cell_type \
    --normalize --zero_inflated --dir_name .
```

---

## Tutorial

The sections below are a condensed reference. For a fully worked walk-through
with the **expected output of every step**, parameter-by-parameter guidance,
and per-step troubleshooting, see the
[online tutorials](https://cytoone.readthedocs.io/en/latest/tutorials/interactive.html).

### Interactive Python

This assumes you are working in a Jupyter notebook or a Python session.

**1. Instantiate the object.** Check your cell-by-marker matrix first so you can
set `normalize` and `zero_inflated` correctly.

```python
from CytoOne.cytoone_class import cytoone

cyto = cytoone(batch_index_col="batch",       # batch column in your metadata
               celltype_col="cell_type",      # cell-type column (plotting only)
               normalize=True,                # arcsinh(./5) if raw-scale
               zero_inflated=True,            # clip negatives to 0 if True
               dr=True)                       # set False to skip the reference UMAP
```

**2. Import data.** This loads and curates the data into the object.

```python
cyto.import_data(cell_by_gene="PATH/TO/CELL-BY-GENE",
                 cell_metadata="PATH/TO/META")
```

> You may pass already-loaded DataFrames instead of paths. CytoOne deep-copies
> them, which increases memory usage.

**3. Train the model.**

```python
cyto.initialize_parameters()
cyto.training_loop()
```

Useful training arguments:

- `n_epoches` — number of epochs (default `50`).
- `n_strata` — number of minibatches per epoch (default `100`).
- `early_stop_pval` — from the 3rd epoch on, CytoOne runs a KS-test on the
  reconstruction loss; once the p-value exceeds this threshold training stops.
  Disabled by default (`early_stop_pval=1`).

**4. Downstream analyses** are all served by `infer()`:

- *Dimension reduction* — embedding for the current data:

  ```python
  _, z_samples = cyto.infer()
  ```

  Or project a new dataset through the trained model:

  ```python
  _, z_samples = cyto.infer(new_cell_by_gene="PATH/TO/NEW",
                            new_cell_metadata="PATH/TO/NEW/META")
  ```

  `z_samples` is a 2-column DataFrame ready for plotting.

- *Batch correction* — normalize every sample to a reference batch:

  ```python
  x_samples, z_samples = cyto.infer(target_batch_index=0)
  ```

  `x_samples` holds the batch-corrected measurements.

- *Differential expression* — draw from the normal component of QZIPN:

  ```python
  x_samples, _ = cyto.infer(get_normal_component=True)
  ```

**5. Save the model.**

```python
cyto.save_model(dir_name="PATH/TO/DIRECTORY", model_name="cyto")
```

This writes `cyto.pt` and `cyto_meta.json`.

**6. Reload later.** Because not all information is stored with the model, you
re-import the data after loading:

```python
from CytoOne.cytoone_class import cytoone

cyto = cytoone(batch_index_col="batch", celltype_col="cell_type",
               normalize=True, zero_inflated=True, model_device="cpu")
cyto.load_model(dir_name="PATH/TO/DIRECTORY", model_name="cyto")
cyto.import_data(cell_by_gene="PATH/TO/CELL-BY-GENE",
                 cell_metadata="PATH/TO/META")
x_samples, z_samples = cyto.infer()
```

### Command-line interface (CLI)

The CLI arguments mirror the Python API.

```shell
python -m CytoOne \
    --batch_index_col batch \
    --celltype_col cell_type \
    --cell_by_gene PATH/TO/CELL-BY-GENE \
    --cell_metadata PATH/TO/META \
    --normalize \
    --zero_inflated \
    --n_epoches 50 \
    --dir_name . \
    --model_name cyto
```

List every option with:

```shell
python -m CytoOne -h
```

---

## Outputs at a glance

| Step | What you get |
|------|--------------|
| `infer()` | `z_samples`: 2-D embedding (`z0`, `z1`) for every cell |
| `infer(target_batch_index=k)` | `x_samples`: measurements normalized to batch `k` (plus `source_batch_index` / `batch_index` columns) |
| `infer(get_normal_component=True)` | `x_samples`: posterior draws from the QZIPN normal component, used for differential analysis |
| `save_model()` | `<name>.pt` (weights) and `<name>_meta.json` (metadata) |
| CLI run | additionally writes `<name>_x_samples.csv` and `<name>_z_samples.csv` |

---

## Troubleshooting

A few of the most common issues (see the
[full troubleshooting guide](https://cytoone.readthedocs.io/en/latest/troubleshooting.html)
for more):

- **Reconstructed values look saturated or all-zero.** Re-check `normalize` and
  `zero_inflated` against your data scale — these two flags must match how your
  measurements were prepared.
- **`KeyError` on a metadata column.** `batch_index_col` / `celltype_col` must
  exactly match the column names in `cell_metadata`.
- **Out-of-memory during import.** Pass file paths rather than in-memory
  DataFrames so CytoOne avoids deep-copying them.
- **Loaded model errors on `infer()`.** Remember to call `import_data()` again
  after `load_model()`.

---

## Citation

If you use CytoOne in your work, a citation is appreciated:

```bibtex
@article{cytoone,
  title   = {Unified Probabilistic Analysis of CyTOF:
             A Deep Generative Approach using CytoOne},
  author  = {Yang, Yuqiu and Wang, Kaiwen and Shen, Yike and
             Weidanz, Jon A and Xiao, Guanghua and Wang, Xinlei},
  year    = {2025},
  note    = {Code and data: https://github.com/Yuqiu-Yang/CytoOne;
             archived at https://zenodo.org/records/17795487}
}
```
