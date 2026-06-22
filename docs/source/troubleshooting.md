# Troubleshooting

A consolidated list of issues users hit, organized by the step where they
appear. If your problem isn't here, please
[open an issue](https://github.com/Yuqiu-Yang/CytoOne/issues).

## Installation

:::{dropdown} `pip install CytoOne` fails to resolve dependencies
:open:
CytoOne pins `python>=3.9,<3.11` and `torch<2.0`. On Python 3.11+ these pins
won't resolve. Create a 3.9 or 3.10 environment:

```shell
conda create -n cytoone python=3.10 && conda activate cytoone
pip install CytoOne
```

Alternatively, use the {ref}`container <install-container>`, which fixes the
interpreter for you.
:::

:::{dropdown} The package installs but `import torch` is very slow / huge
The default PyPI torch wheel bundles CUDA. For a CPU-only, much smaller install,
or to deploy on a server without a GPU, use the Docker image (built CPU-only by
default) — see {doc}`installation`.
:::

:::{dropdown} `ImportError` for scanpy/umap when `dr=True`
The reference UMAP needs scanpy's UMAP stack. Reinstall scanpy
(`pip install "scanpy<1.11"`), or set `dr=False` if you only need CytoOne's own
2-D embedding.
:::

## Importing data

:::{dropdown} `KeyError: '<your column>'` during `import_data`
:open:
`batch_index_col` / `celltype_col` must match the column **names** in your
metadata exactly. Check them:

```python
import pandas as pd
print(pd.read_csv("meta.csv", index_col=0).columns.tolist())
```
:::

:::{dropdown} `n_batches` is 1 when you expected more
Either `cell_metadata` wasn't passed, or the batch column didn't read as
expected. Inspect:

```python
print(cyto.adata.obs["batch"].unique())
print(cyto.adata.uns["n_batches"])
```
:::

:::{dropdown} Cell IDs don't line up between matrix and metadata
Both files must use the **first column** as the cell ID, and the IDs must
match. Mismatched IDs lead to misaligned or dropped rows.
:::

:::{dropdown} `import_data` uses a lot of memory
Pass **file paths** rather than in-memory DataFrames — CytoOne deep-copies any
DataFrame it receives. For very large data, also consider `dr=False` to skip the
reference UMAP.
:::

## Training

:::{dropdown} The epoch counter runs from 0 twice
:open:
Expected. `training_loop` runs a **pretraining** phase followed by the **main**
phase, each up to `n_epoches`.
:::

:::{dropdown} The loss is NaN or explodes
Most often a likelihood/scale mismatch:
- Confirm `normalize` and `zero_inflated` match your data ({doc}`input_format`).
- If using `distribution_type="log_normal"`, switch to the default
  `"softplus_normal"`, which is numerically more stable.
- Try more strata (`n_strata`) so minibatches are smaller.
:::

:::{dropdown} The loss never flattens
Increase `n_epoches`. To stop automatically once it plateaus, set
`early_stop_pval` to e.g. `0.1`.
:::

:::{dropdown} Training is slow on CPU
Use a CUDA build of the container (`--build-arg TORCH_INDEX_URL=...cu117`) and
run with `--gpus all`, or pass `model_device="cuda"`. See {doc}`installation`.
:::

## Inference / downstream

:::{dropdown} Batches still separate after correction
Raise `top_gamma` (default `2.0`) to strengthen MMD-based batch removal, and
make sure you passed `target_batch_index`. Confirm batch info was actually used
by comparing embeddings trained with vs. without `batch_index_col`.
:::

:::{dropdown} Real biological differences look erased after correction
`top_gamma` may be too high — lower it. Batch correction always trades off
against biological signal.
:::

:::{dropdown} The embedding looks "smeared" locally
By design: CytoOne regularizes the latent space toward a Gaussian, favoring
global structure and cluster separability over exact local distances. Use it for
population-level structure, not fine local geometry.
:::

:::{dropdown} Differential-expression Bayes factors are unstable
Posterior draws are stochastic. Average over several
`infer(get_normal_component=True)` draws per condition for stable estimates.
:::

## Saving / loading

:::{dropdown} `infer()` errors right after `load_model()`
:open:
You must call `import_data()` **again** after loading — the data are not stored
with the model. Also reuse the **same** constructor flags (`normalize`,
`zero_inflated`, column names) you trained with.
:::

:::{dropdown} Loading a GPU-trained model on a CPU machine
Construct with `model_device="cpu"` before calling `load_model`.
:::
