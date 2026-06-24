# Running CytoOne in a container

Many research clusters restrict installing software directly onto shared
infrastructure. To make CytoOne easy to deploy in those environments we ship a
ready-to-use container. You can either **pull a pre-built image** or **build it
yourself** from the [`Dockerfile`](../Dockerfile) at the repository root.

The image is CPU-only by default (small, GPU-free, runs anywhere) and bundles
the small example dataset under `/opt/CytoOne/tests`.

---

## 1. Get the image

### Option A — pull the pre-built image (recommended)

Published automatically to the GitHub Container Registry on every push to
`main` (see [`docker-publish.yml`](../.github/workflows/docker-publish.yml)):

```bash
docker pull ghcr.io/yuqiu-yang/cytoone:latest
```

### Option B — build locally

```bash
git clone https://github.com/Yuqiu-Yang/CytoOne.git
cd CytoOne
docker build -t cytoone:latest .
```

Verify the build:

```bash
docker run --rm cytoone:latest --version
# 0.0.2
```

---

## 2. Run an analysis

The default entry point is the CytoOne command-line interface, so anything you
would pass to `python -m CytoOne` can be passed straight to the container.

Mount the folder that holds your data to `/work` (the container's working
directory) so CytoOne can read your inputs and write its outputs back to your
machine:

```bash
docker run --rm -v "$PWD":/work cytoone:latest \
    --cell_by_gene counts.csv \
    --cell_metadata meta.csv \
    --batch_index_col batch \
    --celltype_col cell_type \
    --normalize \
    --zero_inflated \
    --n_epoches 50 \
    --dir_name . \
    --model_name cyto
```

This writes `cyto.pt`, `cyto_meta.json`, `cyto_x_samples.csv` and
`cyto_z_samples.csv` into your current directory.

Try it immediately on the bundled example data:

```bash
docker run --rm cytoone:latest \
    --cell_by_gene /opt/CytoOne/tests/test_data_zi.csv \
    --cell_metadata /opt/CytoOne/tests/test_data_meta.csv \
    --batch_index_col batch --celltype_col cell_type \
    --normalize --zero_inflated --n_epoches 5 --dir_name /work
```

---

## 3. Other run modes

The container ships a small dispatcher (`docker/entrypoint.sh`) with a few
convenience modes:

| Command | What it does |
|---------|--------------|
| `docker run --rm cytoone:latest --help` | CytoOne CLI help (default mode) |
| `docker run --rm -it cytoone:latest python` | Interactive Python with CytoOne importable |
| `docker run --rm -it cytoone:latest bash` | A shell inside the container |
| `docker run --rm -p 8888:8888 -v "$PWD":/work cytoone:latest notebook` | JupyterLab at <http://localhost:8888> for the interactive tutorial |

Example — open the interactive Python API:

```bash
docker run --rm -it -v "$PWD":/work cytoone:latest python
>>> from CytoOne.cytoone_class import cytoone
```

---

## 4. GPU builds (optional)

The default image installs CPU-only PyTorch. To build a CUDA-enabled image,
override the torch wheel index at build time (pick the tag matching your CUDA
toolkit from <https://download.pytorch.org/whl>):

```bash
docker build \
    --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu117 \
    -t cytoone:gpu .

docker run --rm --gpus all cytoone:gpu --help
```

CytoOne automatically uses the GPU when one is visible (`model_device` is set
to `cuda` when available).

---

## 5. Apptainer / Singularity (HPC)

On clusters that provide Apptainer (formerly Singularity) instead of Docker,
convert the published image to a `.sif` file with no Docker daemon required:

```bash
# Easiest: pull and convert in one step
apptainer pull cytoone.sif docker://ghcr.io/yuqiu-yang/cytoone:latest

# Or build from the provided definition file
apptainer build cytoone.sif docker/Singularity.def
```

Run it (bind-mount your data directory with `--bind`):

```bash
apptainer run cytoone.sif --version

apptainer run --bind "$PWD":/work cytoone.sif \
    --cell_by_gene /work/counts.csv \
    --cell_metadata /work/meta.csv \
    --batch_index_col batch --celltype_col cell_type \
    --normalize --zero_inflated --dir_name /work
```

Replace `apptainer` with `singularity` on older installations — the commands
are otherwise identical.

---

## 6. Reproducibility notes

- Every published image is also tagged with its Git commit (`sha-<short-sha>`)
  and, on releases, the version tag (e.g. `v0.0.2`). Pin to one of these rather
  than `latest` when you need an exact, citable environment for a manuscript.
- The base image is `python:3.10-slim`; CytoOne is tested on Python 3.9 and
  3.10.
