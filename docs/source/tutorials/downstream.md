# Downstream analyses

All of CytoOne's downstream capabilities flow from a single trained model and
the `infer()` method. This page covers the three tasks in more depth, each with
the expected output shape. It assumes you have a trained `cyto` object (see the
{doc}`interactive`).

```{contents} On this page
:local:
:depth: 1
```

## Dimension reduction & visualization

The top-level (2-D) latent layer **is** the embedding — no external UMAP/t-SNE
step is needed.

```python
_, z_samples = cyto.infer()
```

`z_samples` is a `(n_cells, 2)` DataFrame with columns `z0`, `z1`. Color by any
per-cell annotation to inspect structure:

```python
import seaborn as sns

z_samples["cell_type"] = cyto.adata.obs["cell_type"].values
sns.scatterplot(data=z_samples, x="z0", y="z1", hue="cell_type", s=8)
```

```{admonition} Interpreting the embedding
:class: tip
CytoOne's latent space is regularized toward a Gaussian, so it favors **global
structure** and **cluster separability** over preserving exact local distances.
Expect clean separation of major populations; do not over-interpret fine local
geometry.
```

Project an unseen dataset through the trained model (useful for applying one
model to new samples):

```python
_, z_new = cyto.infer(
    new_cell_by_gene="./new_counts.csv",
    new_cell_metadata="./new_meta.csv",
)
```

## Batch effect correction

CytoOne models batch effects jointly with the biology, then re-decodes every
cell as if it came from a single **reference batch**.

```python
x_corrected, z_samples = cyto.infer(target_batch_index=0)
```

**Output shape** — `x_corrected` has the marker columns plus two bookkeeping
columns:

```text
['m_0', ..., 'm_9', 'source_batch_index', 'batch_index']
```

- The marker columns hold the **corrected** measurements.
- `source_batch_index` is each cell's original batch.
- `batch_index` is the target batch (all equal to `target_batch_index`).

A simple before/after check — the per-marker distributions of the two batches
should be **closer together after** correction:

```python
import numpy as np

# Before: original (imported) data per batch
before = cyto.adata.to_df()
before["batch"] = cyto.adata.obs["batch_index"].values

# After: corrected data still carries the original batch label
after = x_corrected.drop(columns=["batch_index"]).rename(
    columns={"source_batch_index": "batch"}
)

for m in ["m_0", "m_1"]:
    b0 = before.loc[before.batch == 0, m].mean()
    b1 = before.loc[before.batch == 1, m].mean()
    a0 = after.loc[after.batch == 0, m].mean()
    a1 = after.loc[after.batch == 1, m].mean()
    print(f"{m}: |Δ| before={abs(b0-b1):.3f}  after={abs(a0-a1):.3f}")
```

You should see the after-correction gap shrink relative to before.

```{admonition} Confirming batch info was used
:class: tip
Train once **with** and once **without** `batch_index_col`, then compare the
embeddings colored by batch. With batch information the batches should overlap;
without it they separate — confirming CytoOne removed the batch-driven modes.
```

## Differential expression analysis

CytoOne tests for differential expression using the **normal component** of the
QZIPN likelihood, via a Bayes factor — no reliance on summary statistics, so it
remains sensitive even for heavily zero-inflated markers.

Draw posterior samples of the normal component for each condition:

```python
# condition 1
x1, _ = cyto.infer(new_cell_by_gene=cond1_counts,
                   new_cell_metadata=cond1_meta,
                   get_normal_component=True)

# condition 2
x2, _ = cyto.infer(new_cell_by_gene=cond2_counts,
                   new_cell_metadata=cond2_meta,
                   get_normal_component=True)
```

For a marker `m`, compare the posterior of its normal-component mean between
conditions. Following the paper, the Bayes factor for "marker `m` is higher in
condition 1 than condition 2" is approximated from the posterior samples:

```python
import numpy as np

def bayes_factor(s1, s2):
    """Approximate BF for H1: mean(s1) >= mean(s2) using paired posterior draws."""
    p_h1 = np.mean(s1 >= s2)            # P(H1 | data)
    p_h1 = np.clip(p_h1, 1e-6, 1 - 1e-6)
    return p_h1 / (1 - p_h1)            # equal priors => BF = posterior odds

bf = bayes_factor(x1["m_0"].values, x2["m_0"].values)
print(f"Bayes factor for m_0: {bf:.3f}")
```

A large Bayes factor indicates strong evidence the marker is differentially
expressed between the two conditions. Because the test uses the full posterior
rather than medians, CytoOne can detect differences even when more than half the
raw measurements are zero — a regime where median-based tests report nothing.

```{admonition} Reproducibility
:class: note
Posterior draws are stochastic. For stable Bayes factors, average over several
`infer(get_normal_component=True)` draws per condition (as in the paper).
```

## Summary

| Task | Call | Returns |
|------|------|---------|
| Visualization | `infer()` | `z_samples` (2-D embedding) |
| Batch correction | `infer(target_batch_index=k)` | `x_samples` normalized to batch `k` |
| Differential expression | `infer(get_normal_component=True)` | `x_samples` (normal-component draws) |
```
