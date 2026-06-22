# Interactive Python tutorial

This tutorial walks through a complete CytoOne analysis **step by step**, and
shows the **expected output of each step** so you know what success looks like.
It uses the bundled example data (`tests/test_data_zi.csv` +
`tests/test_data_meta.csv`), so you can follow along verbatim.

```{note}
The numeric values in the "Expected output" blocks below are **illustrative** —
exact losses, timings and RAM figures vary by machine and random seed. What
matters is the **shape** of the output (the lines you see, the DataFrame
columns, the array dimensions).
```

```{contents} On this page
:local:
:depth: 1
```

## Step 0 — Imports

```python
import numpy as np
import pandas as pd
import seaborn as sns
from CytoOne.cytoone_class import cytoone
```

## Step 1 — Instantiate the model

Before anything else, look at your data and decide two things: are the values
raw-scale (set `normalize=True`) and zero-inflated (set `zero_inflated=True`)?
See {doc}`../input_format` if unsure.

```python
cyto = cytoone(
    batch_index_col="batch",        # batch column in the metadata
    celltype_col="cell_type",       # cell-type column (used only for plotting)
    normalize=True,                 # apply arcsinh(./5)
    zero_inflated=True,             # clip negatives to 0
    dr=True,                        # also compute a reference UMAP at import time
)
```

**Expected output:** none — instantiation is silent. `cyto` is now a
`cytoone` object with no data loaded yet (`cyto.adata is None`).

```{admonition} Key constructor arguments
:class: tip
- `latent_dims=[20, 10, 5, 2]` — bottom-up latent layer sizes; the last (`2`)
  is the visualization layer.
- `distribution_type="softplus_normal"` — the QZIPN likelihood. Alternatives:
  `"log_normal"`, `"normal"`.
- `model_device=None` — auto-selects CUDA if available, else CPU.

See {doc}`../parameters` for the full list.
```

## Step 2 — Import the data

```python
cyto.import_data(
    cell_by_gene="./tests/test_data_zi.csv",
    cell_metadata="./tests/test_data_meta.csv",
)
```

**Expected output** — `import_data` reports progress and resource use for each
processing stage (the exact stages depend on your flags):

```text
==============================
Processing cell-by-gene matrix
RAM memory % used: 18.4
RAM Used (GB): 2.713
Done. Time taken is 0.0count seconds.
==============================
==============================
Processing cell metadata
...
==============================
Creating AnnData object
...
==============================
ArcSinh transform
...                                  # only printed because normalize=True
==============================
Clipping the data
...                                  # only printed because zero_inflated=True
==============================
UMAP
...                                  # only printed because dr=True
==============================
```

Confirm the data were read as expected:

```python
print(cyto.adata)
print("markers :", cyto.adata.uns["n_genes"])
print("batches :", cyto.adata.uns["n_batches"])
```

```text
AnnData object with n_obs × n_vars = 4000 × 10
    obs: 'cell_type', 'leiden', 'batch', 'batch_index'
    uns: 'unique_leiden', 'n_leiden', 'cell_type_leiden_map',
         'unique_batch', 'n_batches', 'batch_index_batch_map', 'n_genes',
         'zero_inflated'
    ...
markers : 10
batches : 2
```

```{admonition} Troubleshooting this step
:class: warning
- **`KeyError: 'batch'`** → `batch_index_col` doesn't match a column in your
  metadata. Check `pd.read_csv("meta.csv", index_col=0).columns`.
- **`batches : 1` when you expected more** → your batch column was read as a
  single value, or metadata wasn't passed. Inspect
  `cyto.adata.obs["batch"].unique()`.
- **Values look wrong after import** → re-check `normalize` / `zero_inflated`
  against your data scale.
```

## Step 3 — Initialize parameters

This builds the encoder, decoder, batch embedding, and optimizer.

```python
cyto.initialize_parameters()
```

**Expected output:** none (silent). After this call `cyto.encoder` and
`cyto.decoder` are populated and the model has been moved to `cyto.model_device`.

## Step 4 — Train

```python
cyto.training_loop(n_epoches=50)
```

```{important}
CytoOne trains in **two phases** — a *pretraining* pass followed by the *main*
pass — so the epoch counter runs from `0` to `n_epoches-1` **twice**. This is
expected, not a bug.
```

**Expected output** — periodic progress (every 10th minibatch) and a per-epoch
average:

```text
Train Epoch: 0 [0/100 (0%)]tLoss: 13.812447
Train Epoch: 0 [10/100 (10%)]tLoss: 11.402013
Train Epoch: 0 [20/100 (20%)]tLoss: 10.118755
...
====> Epoch: 0 Average loss: 9.7421
Train Epoch: 1 [0/100 (0%)]tLoss: 8.913220
...
====> Epoch: 49 Average loss: 4.1187
# … then the second (main) phase repeats the epoch counter from 0 …
====> Epoch: 0 Average loss: 5.0023
...
====> Epoch: 49 Average loss: 3.2890
```

The average loss should **trend downward** and then flatten. You can inspect the
recorded losses afterwards:

```python
print("epochs recorded:", len(cyto.RECON_list))
print("last-epoch mean reconstruction loss:", np.mean(cyto.RECON_list[-1]))
```

```{admonition} Tuning training
:class: tip
- `n_epoches` (default 50) — increase if the loss is still falling at the end.
- `n_strata` (default 100) — minibatches per epoch.
- `early_stop_pval` (default 1.0 = off) — set e.g. `0.1` to stop automatically
  once the reconstruction-loss distribution stops changing (KS-test). When it
  triggers you'll see:

  ```text
  ==============================
  No improvement in the reconstruction task detected. Stop early at epoch 23
  ==============================
  ```
```

## Step 5 — Downstream analysis with `infer()`

All three downstream tasks come from the single `infer()` method. The general
signature returns **two** DataFrames: `x_samples` (generated measurements) and
`z_samples` (the 2-D embedding).

### 5a. Dimension reduction / visualization

```python
_, z_samples = cyto.infer()
print(z_samples.shape)
print(z_samples.head())
```

**Expected output** — a `(4000, 2)` DataFrame with columns `z0`, `z1`:

```text
(4000, 2)
                z0        z1
cell_0    1.732...  -0.488...
cell_1    1.690...  -0.451...
cell_2   -0.913...   1.205...
cell_3    1.778...  -0.502...
cell_4   -0.864...   1.147...
```

Plot it, colored by the provided cell types:

```python
z_samples["cell_type"] = cyto.adata.obs["cell_type"].values
sns.scatterplot(data=z_samples, x="z0", y="z1", hue="cell_type", s=8)
```

You should see the two simulated cell types separate into distinct regions.

To project a **new** dataset through the already-trained model:

```python
_, z_new = cyto.infer(
    new_cell_by_gene="./path/to/new_counts.csv",
    new_cell_metadata="./path/to/new_meta.csv",
)
```

### 5b. Batch correction

Normalize every cell to a chosen reference batch (here batch `0`):

```python
x_samples, z_samples = cyto.infer(target_batch_index=0)
print(x_samples.columns.tolist())
print(x_samples.shape)
```

**Expected output** — generated measurements plus two bookkeeping columns:

```text
['m_0', 'm_1', ..., 'm_9', 'source_batch_index', 'batch_index']
(4000, 12)
```

- `source_batch_index` — the cell's original batch.
- `batch_index` — the target batch everything was normalized to (all `0` here).

The marker columns (`m_0` … `m_9`) are the **batch-corrected** measurements.
See {doc}`downstream` for a before/after comparison.

### 5c. Differential expression

Draw from the normal component of the QZIPN likelihood (the quantity CytoOne
uses for differential testing):

```python
x_samples, _ = cyto.infer(get_normal_component=True)
```

`x_samples` now holds posterior samples of the latent normal component for each
marker, which you can compare between conditions. The full Bayes-factor workflow
is described in {doc}`downstream`.

## Step 6 — Save the model

```python
cyto.save_model(dir_name=".", model_name="cyto")
```

**Expected output:** none. Two files appear in `dir_name`:

```text
cyto.pt           # PyTorch weights + optimizer state
cyto_meta.json    # data/encoder/decoder parameters needed to rebuild the model
```

## Step 7 — Reload the model later

Because some information (the data itself) is **not** stored with the model, you
re-import the data after loading.

```python
from CytoOne.cytoone_class import cytoone

cyto = cytoone(
    batch_index_col="batch",
    celltype_col="cell_type",
    normalize=True,
    zero_inflated=True,
    model_device="cpu",     # load on CPU even if trained on GPU
)
cyto.load_model(dir_name=".", model_name="cyto")
cyto.import_data(
    cell_by_gene="./tests/test_data_zi.csv",
    cell_metadata="./tests/test_data_meta.csv",
)
x_samples, z_samples = cyto.infer()
```

```{admonition} Common reload error
:class: warning
**`AttributeError` / shape mismatch on `infer()` after `load_model()`** almost
always means `import_data()` wasn't called again, or the constructor flags
(`normalize`, `zero_inflated`, column names) differ from the original run. Use
the same settings you trained with.
```

## Where to go next

- {doc}`downstream` — worked batch-correction and differential-expression
  examples with expected outputs.
- {doc}`cli` — run the same workflow from the command line.
- {doc}`../parameters` — every parameter and its effect.
- {doc}`../troubleshooting` — a consolidated problem/solution table.
```
