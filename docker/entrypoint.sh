#!/usr/bin/env bash
###############################################################################
# CytoOne container entry point                                               #
#                                                                             #
# Usage patterns:                                                             #
#   docker run --rm cytoone:latest --help            # CytoOne CLI (default)  #
#   docker run --rm cytoone:latest --version                                  #
#   docker run --rm -v "$PWD":/work cytoone:latest \                          #
#       --cell_by_gene counts.csv --cell_metadata meta.csv ...                #
#                                                                             #
#   docker run --rm -it cytoone:latest python        # interactive Python     #
#   docker run --rm -it cytoone:latest bash          # shell                  #
#   docker run --rm -p 8888:8888 -v "$PWD":/work \                            #
#       cytoone:latest notebook                       # JupyterLab            #
###############################################################################
set -euo pipefail

case "${1:-}" in
    notebook|jupyter|lab)
        shift || true
        exec jupyter lab \
            --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
            --ServerApp.root_dir=/work "$@"
        ;;
    python)
        shift
        exec python "$@"
        ;;
    bash|sh)
        shift
        exec /bin/bash "$@"
        ;;
    cli)
        # Explicit CLI mode (equivalent to the default)
        shift
        exec python -m CytoOne "$@"
        ;;
    *)
        # Default: forward everything to the CytoOne CLI
        exec python -m CytoOne "$@"
        ;;
esac
