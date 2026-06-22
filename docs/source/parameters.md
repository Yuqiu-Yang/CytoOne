# Parameter guide

This page lists CytoOne's user-facing parameters, their defaults, and **how
changing them affects your results**. For the full machine-generated signatures
see the {doc}`api`.

```{contents} On this page
:local:
:depth: 1
```

## Constructor — `cytoone(...)`

### Data-handling flags

| Parameter | Default | Effect of changing it |
|-----------|---------|-----------------------|
| `batch_index_col` | `None` | Name of the batch column. **Leave `None`** for single-batch data (batch correction is then a no-op). Setting it enables batch modeling/correction. |
| `celltype_col` | `None` | Name of the cell-type column. **Plotting only** — has no effect on the model fit. |
| `normalize` | `True` | `True` applies `arcsinh(./5)`. Set `False` only if your data are *already* transformed; otherwise the likelihood is mismatched and reconstructions degrade. |
| `zero_inflated` | `True` | `True` clips negatives to 0 and models a zero gate (raw-style data). `False` instead models added Gaussian noise (Helios-style). Must match your data's nature. |
| `dr` | `True` | `True` also computes a reference UMAP at import time. Set `False` to **skip UMAP** and speed up `import_data` when you only need CytoOne's own embedding. |

```{admonition} The two flags that matter most
:class: important
`normalize` and `zero_inflated` describe *your data*, not preferences. Getting
them wrong is the single most common cause of poor reconstructions — see
{doc}`input_format`.
```

### Model-architecture parameters

| Parameter | Default | Effect of changing it |
|-----------|---------|-----------------------|
| `latent_dims` | `[20, 10, 5, 2]` | Bottom-up sizes of the hierarchical latent layers. The **last entry is fixed at 2** for visualization. Larger lower layers increase capacity (better reconstruction / correction) at the cost of compute and overfitting risk. |
| `batch_embedding_dim` | `2` | Size of the learned batch embedding. Increase only if you have many batches with complex effects. |
| `encoder_hidden_dims` | nested list | Hidden widths per encoder block. Wider/deeper = more capacity, slower training. |
| `decoder_hidden_dims` | nested list | As above, for the decoder. Keep roughly mirrored with the encoder. |
| `drop_out_p` | `0.2` | Dropout probability. Higher values regularize more (less overfitting, potentially underfitting). |
| `distribution_type` | `"softplus_normal"` | The likelihood. `"softplus_normal"` (QZIPN) is recommended and most stable; `"log_normal"` is numerically less stable; `"normal"` is a plain-Gaussian baseline (loses zero-inflation modeling). |
| `decoupled_gate` | `True` | Whether the zero-inflation gate is modeled separately from the continuous component. |
| `model_device` | `None` | `None` auto-selects CUDA if present, else CPU. Pass `"cpu"` or `"cuda"` to force. Use `"cpu"` when loading a GPU-trained model on a CPU-only machine. |

### Loss-weighting parameters

| Parameter | Default | Effect of changing it |
|-----------|---------|-----------------------|
| `top_beta` | `0.01` | KL weight on the **top** latent layer. Smaller values relax the Gaussian prior on the 2-D layer (sharper clusters, less smoothing); larger values regularize it more. |
| `top_gamma` | `2.0` | MMD weight on the top layer — the **strength of batch-effect removal**. Increase if batches still separate after correction; decrease if biological signal is being washed out. |
| `pretrain_beta` | `1.0` | KL weight used during the pretraining phase. |
| `pretrain_gamma` | `1.0` | MMD weight used during the pretraining phase. |

```{admonition} If batch effects persist
:class: tip
Raise `top_gamma`. If, conversely, real biological differences between batches
are being erased, lower it. The remaining layer weights are scaled
automatically from these top-level values.
```

## Training — `training_loop(...)`

| Parameter | Default | Effect of changing it |
|-----------|---------|-----------------------|
| `n_epoches` | `50` | Epochs **per phase** (CytoOne runs a pretraining phase then a main phase). Increase if the average loss is still falling at the end. |
| `n_strata` | `100` | Minibatches per epoch. Each stratum is guaranteed to contain all batches. More strata = smaller minibatches. |
| `early_stop_pval` | `1.0` | From epoch 3 on, a KS-test compares consecutive epochs' reconstruction-loss distributions. When the p-value exceeds this threshold, training stops. `1.0` disables early stopping; e.g. `0.1` enables it. |

## Inference — `infer(...)`

| Parameter | Default | Effect of changing it |
|-----------|---------|-----------------------|
| `new_cell_by_gene` / `new_cell_metadata` | `None` | Provide to project an **unseen** dataset through the trained model instead of the training data. |
| `target_batch_index` | `None` | `None` keeps each cell in its own batch. An integer normalizes **all** cells to that reference batch (batch correction). |
| `mode` | `"random"` | `"random"` samples the latent posterior; `"fix"` uses the posterior mean (deterministic, less variable embeddings). |
| `denoise` | `False` | `True` returns the de-noised (non-zero-inflated) signal. |
| `use_pretrain` | `False` | Use the pretraining sub-model rather than the full model. |
| `get_normal_component` | `False` | `True` returns posterior draws of the QZIPN **normal component** — the quantity used for differential expression. |

## Optimizer

CytoOne uses Adam with a learning rate of `1e-3` (set internally). The latent
KL weights default to `λ₁ = 0.01` with the remainder scaled by latent
dimension, and the MMD weights default to `γ₁ = 2` scaled inversely with latent
dimension — matching the settings reported in the paper.
