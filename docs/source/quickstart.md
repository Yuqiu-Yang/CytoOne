# Quick start

This page gets you from a fresh install to a trained model and an embedding
using the **bundled example data** — no downloads required.

The example data live in the repository's `tests/` folder and were simulated
with [Cytomulate](https://github.com/kevin931/cytomulate):

| File | Shape | Description |
|------|-------|-------------|
| `tests/test_data_zi.csv` | 4000 × 10 | Zero-inflated marker measurements |
| `tests/test_data_meta.csv` | 4000 × 2 | `batch` (2 batches), `cell_type` (2 types) |

## Python

```python
from CytoOne.cytoone_class import cytoone

# 1. Instantiate
cyto = cytoone(
    batch_index_col="batch",
    celltype_col="cell_type",
    normalize=True,        # arcsinh(./5) — the example data are raw-scale
    zero_inflated=True,    # the *_zi data are zero-inflated
    dr=True,               # also compute a reference UMAP
)

# 2. Import data
cyto.import_data(
    cell_by_gene="./tests/test_data_zi.csv",
    cell_metadata="./tests/test_data_meta.csv",
)

# 3. Train
cyto.initialize_parameters()
cyto.training_loop(n_epoches=50)

# 4. Get a 2-D, batch-effect-free embedding
_, z_samples = cyto.infer()
print(z_samples.head())
```

`z_samples` is a `pandas.DataFrame` with one row per cell and two columns,
`z0` and `z1`:

```text
                z0        z1
cell_0    1.83...   -0.51...
cell_1    1.79...   -0.48...
cell_2   -0.94...    1.22...
...
```

Plot it (colored by the cell-type annotation that came with the data):

```python
import seaborn as sns

z_samples["cell_type"] = cyto.adata.obs["cell_type"].values
sns.scatterplot(data=z_samples, x="z0", y="z1", hue="cell_type", s=8)
```

## Command line

The same run as a single command:

```shell
python -m CytoOne \
    --cell_by_gene ./tests/test_data_zi.csv \
    --cell_metadata ./tests/test_data_meta.csv \
    --batch_index_col batch --celltype_col cell_type \
    --normalize --zero_inflated \
    --n_epoches 50 \
    --dir_name . --model_name cyto
```

This writes four files into the current directory:

```text
cyto.pt                 # trained weights
cyto_meta.json          # model metadata
cyto_x_samples.csv      # generated/normalized measurements
cyto_z_samples.csv      # 2-D embedding
```

## What's next

- {doc}`tutorials/interactive` — the full Python workflow with the expected
  output of **every** step.
- {doc}`tutorials/downstream` — batch correction, differential expression, and
  dimension reduction in detail.
- {doc}`parameters` — what each setting does and when to change it.
```
