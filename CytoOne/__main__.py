"""The Command-Line Interface (CLI) of CytoOne

The CLI of CytoOne can be accessed via ``python -m CytoOne``.

:Example:

    Get help:
    
    .. code-block:: bash

        python -m CytoOne -h
    
    Check version and authors:
    
    .. code-block:: bash
    
        python -m CytoOne --version 
        python -m CytoOne --author

**Pros**
    * Shallow learning curve
    
**Cons**
    * Fewer configurations 
    * No inspections of intermediate results
"""


import os
import sys
import torch
import argparse

from CytoOne.__version__ import __version__, __author__
from CytoOne.cytoone_class import cytoone


parser = argparse.ArgumentParser(description="CytoOne")

parser.add_argument("--version", action="version",
                    version=__version__, help="Display the version of the software")
parser.add_argument("--author", action="version", version=__author__,
                    help="Check the author list of the algorithm")
# Instantiate object 
parser.add_argument("--batch_index_col", nargs="?", type=str,
                    help="The column containing batch information")
parser.add_argument("--celltype_col", nargs="?", type=str,
                    help="The column cell type information")
parser.add_argument("--normalize", action="store_true",
                    help="If normalize the data via asinh")
parser.add_argument("--zero_inflated", action="store_true",
                    help="If the data if zero-inflated")
# Import data 
parser.add_argument("--cell_by_gene", type=str,
                    help="The path to the csv file that contains the CyTOF measurements")
parser.add_argument("--cell_metadata", type=str,
                    help="The path to the csv file that contains metainformation of the CyTOF measurements")
# Model training
parser.add_argument("--n_epoches", type=int, default=50,
                    help="Number of epoches")
parser.add_argument("--n_strata", type=int, default=100,
                    help="Number of minibatches per epoch")
# Downstream analysis 
parser.add_argument("--target_batch_index", type=int, nargs="?",
                    help="Batch to which batch correction is targeted")
parser.add_argument("--get_normal_component", action='store_true',
                    help="Whether to get the normal component")
# Model saving 
parser.add_argument("--dir_name", type=str, default=".",
                    help="The directory path at which the saved model should be.")
parser.add_argument("--model_name", type=str, default="cytoone",
                    help="The model name")



def main(cmdargs: argparse.Namespace):
    """The main method for CytoOne

    Parameters:
    ----------
    cmdargs: argparse.Namespace
        The command line argments and flags 
    """
    batch_index_col = cmdargs.batch_index_col
    celltype_col = cmdargs.celltype_col
    normalize = cmdargs.normalize
    zero_inflated = cmdargs.zero_inflated

    cyto = cytoone(batch_index_col=batch_index_col,
                   celltype_col=celltype_col,
                   normalize=normalize,
                   dr=False,
                   zero_inflated=zero_inflated)
    
    cell_by_gene = cmdargs.cell_by_gene 
    cell_metadata = cmdargs.cell_metadata
    
    cyto.import_data(cell_by_gene=cell_by_gene,
                     cell_metadata=cell_metadata)
    
    cyto.initialize_parameters()

    n_epoches = cmdargs.n_epoches
    n_strata = cmdargs.n_strata

    cyto.training_loop(n_epoches=n_epoches,
                       n_strata=n_strata)
    
    target_batch_index = cmdargs.target_batch_index
    get_normal_component = cmdargs.get_normal_component

    x_samples, z_samples = cyto.infer(target_batch_index=target_batch_index,
                                        get_normal_component=get_normal_component)

    dir_name = cmdargs.dir_name 
    model_name = cmdargs.model_name 
    cyto.save_model(dir_name=dir_name,
                    model_name=model_name)
    x_samples.to_csv(os.path.join(dir_name, model_name+"_x_samples.csv"),
                     index=True)
    z_samples.to_csv(os.path.join(dir_name, model_name+"_z_samples.csv"),
                     index=True)

    sys.exit(0)


if __name__ == "__main__":
    cmdargs = parser.parse_args()
    main(cmdargs=cmdargs)