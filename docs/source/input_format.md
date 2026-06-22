# Input format

CytoOne accepts up to two inputs. Both may be supplied either as a path to a
`.csv` file or as an in-memory `pandas.DataFrame`.

## 1. Cell-by-marker matrix (`cell_by_gene`)

A table with **one row per cell** and **one column per protein marker**.

- The **first column is the cell ID** (used as the row index).
- Remaining column names are marker names.
- Values are protein measurements.

Example (`tests/test_data_zi.csv`):

```text
        ,m_0   ,m_1   ,m_2   , ... ,m_9
cell_0  ,0.58  ,3.62  ,1.55  , ... ,2.32
cell_1  ,0.00  ,1.95  ,3.24  , ... ,0.13
cell_2  ,0.00  ,0.00  ,2.71  , ... ,1.04
...
```

### Scaling: `normalize` and `zero_inflated`

These two flags must match how your measurements were prepared.

| Flag | When `True` | When `False` |
|------|-------------|--------------|
| `normalize` | CytoOne applies `arcsinh(x / 5)` for you | data are assumed already `arcsinh`-transformed |
| `zero_inflated` | negative values are clipped to 0 (raw-style data with true zeros) | a small Gaussian-noise term is modeled instead (Helios-style noisy data) |

```{tip}
After `arcsinh(./5)` transformation, marker values are typically in the **0–10**
range. If you see values in the **hundreds**, your data are still raw — keep
`normalize=True`. If they are already in the 0–10 range, set `normalize=False`.
```

## 2. Cell metadata (`cell_metadata`, optional)

Per-cell annotations. As with the matrix, the **first column is the cell ID**
and must match the IDs in `cell_by_gene`.

Recognized columns (you choose which by name):

| Role | Set via | Used for |
|------|---------|----------|
| Cell ID | first column (index) | aligning cells to the matrix |
| Batch | `batch_index_col` | batch-effect modeling & correction |
| Cell type | `celltype_col` | **plotting only** — never used by the model |

Example (`tests/test_data_meta.csv`):

```text
        ,batch ,cell_type
cell_0  ,1     ,1
cell_1  ,1     ,1
cell_2  ,2     ,2
...
```

```{note}
Your column names need not be `batch` / `cell_type`. Pass whatever they are
called, e.g. `batch_index_col="experiment_id"`,
`celltype_col="annotated_population"`.
```

### If you omit the metadata

If `cell_metadata` is not provided, CytoOne assumes **all cells belong to a
single batch** and skips batch correction. This is fine for single-batch
datasets or for a quick visualization run.

## Passing DataFrames instead of paths

```python
import pandas as pd

counts = pd.read_csv("counts.csv", index_col=0)
meta = pd.read_csv("meta.csv", index_col=0)

cyto.import_data(cell_by_gene=counts, cell_metadata=meta)
```

```{warning}
CytoOne deep-copies any DataFrame you pass in, which increases peak memory use.
For large datasets, prefer passing **file paths** so CytoOne reads them directly.
```

## What `import_data` produces

Internally, `import_data` builds an
[`AnnData`](https://anndata.readthedocs.io) object stored at `cyto.adata`. It
records, among other things:

- `cyto.adata.uns["n_genes"]` — number of markers,
- `cyto.adata.uns["n_batches"]` — number of distinct batches,
- `cyto.adata.obs["batch_index"]` — integer batch code per cell,
- `cyto.adata.obs["cell_type"]` — the (optional) cell-type label.

You can inspect these at any point to confirm the data were read as expected
(see {doc}`troubleshooting`).
