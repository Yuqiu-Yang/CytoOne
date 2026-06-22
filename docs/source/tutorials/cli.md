# Command-line tutorial

The CLI runs the full CytoOne workflow — import → train → infer → save — in a
single command. It is the easiest way to use CytoOne inside a
{ref}`container <install-container>` or a batch/cluster job.

Its arguments mirror the Python API.

## Help and version

```shell
python -m CytoOne -h
```

**Expected output** (abridged):

```text
usage: CytoOne [-h] [--version] [--author] [--batch_index_col [BATCH_INDEX_COL]]
               [--celltype_col [CELLTYPE_COL]] [--normalize] [--zero_inflated]
               [--cell_by_gene CELL_BY_GENE] [--cell_metadata CELL_METADATA]
               [--n_epoches N_EPOCHES] [--n_strata N_STRATA]
               [--target_batch_index [TARGET_BATCH_INDEX]] [--get_normal_component]
               [--dir_name DIR_NAME] [--model_name MODEL_NAME]

CytoOne
...
```

```shell
python -m CytoOne --version
# 0.0.2

python -m CytoOne --author
# Yuqiu Yang; Kaiwen Wang; Xinlei Wang
```

## A full run

```shell
python -m CytoOne \
    --cell_by_gene ./tests/test_data_zi.csv \
    --cell_metadata ./tests/test_data_meta.csv \
    --batch_index_col batch \
    --celltype_col cell_type \
    --normalize \
    --zero_inflated \
    --n_epoches 50 \
    --n_strata 100 \
    --dir_name . \
    --model_name cyto
```

**Expected output** — the same import logs and training progress as the Python
API (training runs a pretraining phase then a main phase):

```text
==============================
Processing cell-by-gene matrix
...
==============================
Train Epoch: 0 [0/100 (0%)]tLoss: 13.81...
...
====> Epoch: 49 Average loss: 3.28...
```

On completion, four files are written to `--dir_name`:

```text
cyto.pt                 # trained weights + optimizer state
cyto_meta.json          # model metadata
cyto_x_samples.csv      # generated measurements (markers + batch columns)
cyto_z_samples.csv      # 2-D embedding (z0, z1) per cell
```

## Argument reference

| Flag | Maps to | Default | Notes |
|------|---------|---------|-------|
| `--cell_by_gene` | `import_data(cell_by_gene=...)` | — | path to the marker matrix CSV |
| `--cell_metadata` | `import_data(cell_metadata=...)` | — | path to the metadata CSV |
| `--batch_index_col` | constructor | `None` | batch column name |
| `--celltype_col` | constructor | `None` | cell-type column name (plotting only) |
| `--normalize` | constructor | off | flag; apply `arcsinh(./5)` |
| `--zero_inflated` | constructor | off | flag; clip negatives to 0 |
| `--n_epoches` | `training_loop` | `50` | epochs per phase |
| `--n_strata` | `training_loop` | `100` | minibatches per epoch |
| `--target_batch_index` | `infer` | `None` | reference batch for correction |
| `--get_normal_component` | `infer` | off | flag; sample the QZIPN normal component |
| `--dir_name` | `save_model` / output CSVs | `.` | output directory |
| `--model_name` | `save_model` / output CSVs | `cytoone` | output filename stem |

```{note}
`--normalize`, `--zero_inflated` and `--get_normal_component` are **flags** — their
mere presence sets them to `True`. Omit them to keep the value `False`.
```

## Batch correction from the CLI

Normalize everything to batch `0` and write the corrected measurements:

```shell
python -m CytoOne \
    --cell_by_gene ./tests/test_data_zi.csv \
    --cell_metadata ./tests/test_data_meta.csv \
    --batch_index_col batch --celltype_col cell_type \
    --normalize --zero_inflated \
    --target_batch_index 0 \
    --dir_name . --model_name cyto_bc
```

The corrected measurements are in `cyto_bc_x_samples.csv`.

## Running inside the container

Identical command, just prefixed with the container runtime and a mounted data
directory:

```shell
docker run --rm -v "$PWD":/work cytoone:latest \
    --cell_by_gene /opt/CytoOne/tests/test_data_zi.csv \
    --cell_metadata /opt/CytoOne/tests/test_data_meta.csv \
    --batch_index_col batch --celltype_col cell_type \
    --normalize --zero_inflated --dir_name /work
```

See {doc}`../installation` for Apptainer/Singularity equivalents.
```
