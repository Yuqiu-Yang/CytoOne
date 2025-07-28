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


