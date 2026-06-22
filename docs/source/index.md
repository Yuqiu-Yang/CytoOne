# CytoOne

> A unified probabilistic framework for CyTOF data

CytoOne is a single, coherent Bayesian deep generative model for mass cytometry
(CyTOF) data. Rather than chaining together task-specific tools — each with its
own, sometimes conflicting, distributional assumptions — CytoOne performs
**batch effect correction**, **differential expression analysis**, and
**visualization / dimension reduction** within one model trained end-to-end.

It pairs a hierarchical latent architecture inspired by the Nouveau Variational
Autoencoder (NVAE) with a purpose-built **quasi zero-inflated softplus-normal
(QZIPN)** likelihood tailored to the sparse, noisy nature of CyTOF
measurements.

```{image} https://raw.githubusercontent.com/Yuqiu-Yang/CytoOne/main/assets/model_overview.png
:alt: CytoOne model overview
:align: center
```

---

## Get started

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🛠️ Installation
:link: installation
:link-type: doc

conda + pip, building from source, or the install-free Docker / Apptainer
container for shared servers.
:::

:::{grid-item-card} 🚀 Quick start
:link: quickstart
:link-type: doc

Train CytoOne on the bundled example data and get an embedding in a couple of
minutes.
:::

:::{grid-item-card} 📦 Input format
:link: input_format
:link-type: doc

Exactly what the cell-by-marker matrix and metadata should look like.
:::

:::{grid-item-card} 📚 Tutorials
:link: tutorials/interactive
:link-type: doc

Step-by-step walk-throughs with the expected output of every step.
:::

:::{grid-item-card} 🎛️ Parameter guide
:link: parameters
:link-type: doc

What each knob does and how changing it affects your results.
:::

:::{grid-item-card} 🩺 Troubleshooting
:link: troubleshooting
:link-type: doc

Symptoms, causes, and fixes for the issues users hit most.
:::

::::

---

## How CytoOne fits a typical analysis

1. **Instantiate** a `cytoone` object, telling it how your data are scaled
   (`normalize`, `zero_inflated`) and which metadata columns to use.
2. **Import** your cell-by-marker matrix and (optionally) per-cell metadata.
3. **Train** with stochastic variational inference until the reconstruction loss
   stabilizes.
4. **Infer** — a single `infer()` call serves visualization, batch correction,
   and differential expression.
5. **Save / reload** the trained model for later reuse.

See the {doc}`interactive tutorial <tutorials/interactive>` for the full flow
with outputs, or the {doc}`CLI tutorial <tutorials/cli>` for the one-command
version.

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
input_format
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/interactive
tutorials/cli
tutorials/downstream
```

```{toctree}
:hidden:
:caption: Reference

parameters
troubleshooting
api
```

```{toctree}
:hidden:
:caption: Links

GitHub repository <https://github.com/Yuqiu-Yang/CytoOne>
Archived data (Zenodo) <https://zenodo.org/records/17795487>
```
