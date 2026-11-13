# Installation

## Practical hardware guidance 

The scalability results reported in the manuscript were obtained on a server node equipped with an Intel Xeon Gold 6240 host, 384 GB of system memory, and an NVIDIA V100 GPU with 32 GB of device memory. Absolute runtimes and memory usage are hardware-specific; the benchmark is intended primarily to show scaling with cell number and marker-panel size.

We additionally benchmarked CPU-only CytoOne training on an Apple M3 laptop with 16 GB of memory. Runtime increased from 17.8 seconds at 100,000 cells to 290.6 seconds at 2,000,000 cells, with peak process memory ranging from 1.18 to 3.12 GB. Full model fitting requires multiple epochs and will take proportionally longer. 

### Approximate guidance 

- Small datasets (~100,000 cells): readily handled by the tested CPU-only laptop configuration.
- Medium datasets (~500,000-1 million cells): feasible on a 16-GB laptop, although full fitting may require substantial wall time.
- Very large datasets (~1.5-2 million cells): also completed within 16 GB on the tested M3 laptop; GPU or server hardware is recommended when repeated fits or rapid turnaround are required.

CytoOne is tested on **Python 3.9 and 3.10**. Choose whichever route fits your
environment — for shared / locked-down servers, jump to the
{ref}`container option <install-container>`.

## Option 1 — conda + pip

Create and activate a clean environment:

```shell
conda create -n cytoone python=3.10
conda activate cytoone
```

Install the stable release from PyPI:

```shell
pip install CytoOne
```

Check it worked:

```shell
python -m CytoOne --version
```

Expected output:

```text
0.0.2
```

## Option 2 — build from source

The newest features are on GitHub. Clone, build a wheel, and install it:

```shell
git clone https://github.com/Yuqiu-Yang/CytoOne.git
cd CytoOne
python setup.py sdist bdist_wheel
```

A `dist/` directory now contains the wheel. Install it (substitute the actual
`VERSION` you see in `dist/`):

```shell
cd ./dist
pip install ./CytoOne-VERSION-py3-none-any.whl
```

(install-container)=
## Option 3 — Docker / Apptainer (recommended for shared servers)

Many institutions restrict installing software directly onto common
infrastructure. The container removes that friction: nothing is installed on the
host, and the environment is identical for every user.

### Docker

Pull the pre-built image from the GitHub Container Registry:

```shell
docker pull ghcr.io/yuqiu-yang/cytoone:latest
docker run --rm ghcr.io/yuqiu-yang/cytoone:latest --version
```

Or build it yourself from the `Dockerfile` at the repository root:

```shell
docker build -t cytoone:latest .
```

Run an analysis, mounting your data folder to `/work`:

```shell
docker run --rm -v "$PWD":/work cytoone:latest \
    --cell_by_gene counts.csv \
    --cell_metadata meta.csv \
    --batch_index_col batch --celltype_col cell_type \
    --normalize --zero_inflated --dir_name .
```

Launch the interactive tutorial in JupyterLab:

```shell
docker run --rm -p 8888:8888 -v "$PWD":/work cytoone:latest notebook
```

### Apptainer / Singularity (HPC)

On clusters without Docker, convert the same image to a `.sif` file:

```shell
# Pull and convert in one step
apptainer pull cytoone.sif docker://ghcr.io/yuqiu-yang/cytoone:latest

# …or build from the bundled definition file
apptainer build cytoone.sif docker/Singularity.def
```

Run it, binding your working directory:

```shell
apptainer run --bind "$PWD":/work cytoone.sif \
    --cell_by_gene /work/counts.csv \
    --cell_metadata /work/meta.csv \
    --batch_index_col batch --celltype_col cell_type \
    --normalize --zero_inflated --dir_name /work
```

### GPU builds

The default image ships CPU-only PyTorch. For a CUDA build, override the wheel
index at build time:

```shell
docker build \
    --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu117 \
    -t cytoone:gpu .
docker run --rm --gpus all cytoone:gpu --help
```

Full container documentation lives in
[`docker/README.md`](https://github.com/Yuqiu-Yang/CytoOne/blob/main/docker/README.md).

## Dependencies

These are installed automatically; listed here for reference.

| Package | Constraint |
|---------|------------|
| python | `>=3.9,<3.11` |
| numpy | `<2.0` |
| pandas | `>=2.2.0` |
| anndata | `>=0.10,<0.11` |
| scanpy | `<1.11` |
| torch | `<2.0` |
| pyro-ppl | `<1.8.5` |
| seaborn | — |
| jupyter | — |
| ipywidgets | — |

## Verifying the installation

```shell
python -m CytoOne --version    # prints the version
python -m CytoOne --author     # prints the author list
python -m CytoOne -h           # lists every CLI option
```

If these run without error, you're ready for the {doc}`quickstart`.
