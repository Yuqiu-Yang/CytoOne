# syntax=docker/dockerfile:1
###############################################################################
# CytoOne — reproducible analysis container                                   #
# ----------------------------------------------------------------------------#
# A unified probabilistic framework for CyTOF data.                           #
#                                                                             #
# Quick start                                                                 #
#   docker build -t cytoone:latest .                                          #
#   docker run --rm cytoone:latest --version                                  #
#   docker run --rm cytoone:latest --help                                     #
#                                                                             #
# See docker/README.md for full usage, GPU builds, mounting your own data,    #
# launching JupyterLab, and Apptainer / Singularity conversion.               #
###############################################################################

FROM python:3.10-slim AS base

# ---- OCI image metadata (shown by `docker inspect`) ------------------------
LABEL org.opencontainers.image.title="CytoOne" \
      org.opencontainers.image.description="A unified probabilistic framework for CyTOF data" \
      org.opencontainers.image.source="https://github.com/Yuqiu-Yang/CytoOne" \
      org.opencontainers.image.documentation="https://cytoone.readthedocs.io" \
      org.opencontainers.image.licenses="BSD-3-Clause"

# ---- Sensible Python / pip defaults ----------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- Minimal runtime system libraries --------------------------------------
# libgomp1 provides the OpenMP runtime needed by numpy / scikit-learn /
# pynndescent (used by scanpy's UMAP). We avoid a compiler toolchain because
# every pinned dependency ships pre-built wheels for CPython 3.10.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/CytoOne

# ---- Torch flavour ---------------------------------------------------------
# CPU-only by default: small image, runs on any server, no GPU drivers needed.
# For a CUDA-enabled image rebuild with e.g.
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu117 -t cytoone:gpu .
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Install torch first, in its own layer, so the large wheel is cached and not
# re-downloaded whenever the package source changes. The pinned "<2.0" matches
# requirements.txt, so the later requirements install will not replace it.
RUN pip install --index-url ${TORCH_INDEX_URL} "torch<2.0"

# ---- Python dependencies (cached separately from the source) ---------------
COPY requirements.txt ./
RUN pip install -r requirements.txt

# ---- Install CytoOne itself ------------------------------------------------
COPY . .
RUN pip install . \
    && python -c "import CytoOne; print('CytoOne', CytoOne.__version__.__version__, 'installed OK')"

# The bundled example dataset is available inside the image at:
#   /opt/CytoOne/tests/test_data_zi.csv
#   /opt/CytoOne/tests/test_data_n.csv
#   /opt/CytoOne/tests/test_data_meta.csv

# ---- Entry point -----------------------------------------------------------
# A thin dispatcher lets users pick a mode:
#   (default)            -> `python -m CytoOne ...`   (CytoOne CLI)
#   notebook | jupyter   -> JupyterLab on port 8888
#   python | bash        -> a Python shell / a shell
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# /work is the default location for user-mounted data:
#   docker run --rm -v "$PWD":/work cytoone:latest --cell_by_gene my_counts.csv ...
WORKDIR /work

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
