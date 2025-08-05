import pytest
import numpy as np
import pandas as pd 
from CytoOne.cytoone_class import cytoone

counts_zi = pd.read_csv("./tests/test_data_zi.csv", index_col=0)
counts_zi.index.rename(name='cell_id', inplace=True)
counts_n = pd.read_csv("./tests/test_data_n.csv", index_col=0)
counts_n.index.rename(name='cell_id', inplace=True)
meta = pd.read_csv("./tests/test_data_meta.csv", index_col=0)
meta.index.rename(name='cell_id', inplace=True)


@pytest.mark.parametrize("batch_index_col, normalize, dr, zero_inflated, distribution_type, cell_by_gene, cell_metadata, target_batch_index, get_normal_component",
                         [
                             ("batch", True, True, True, "softplus_normal", counts_zi, meta, None, False),
                             ("batch", True, True, True, "log_normal", counts_zi, meta, None, False),
                             ("batch", True, True, True, "normal", counts_zi, meta, None, False),
                         ])
def test_cytoone(batch_index_col,
                 normalize,
                 dr,
                 zero_inflated,
                 distribution_type,
                 cell_by_gene,
                 cell_metadata,
                 target_batch_index,
                 get_normal_component):
    cyto = cytoone(batch_index_col=batch_index_col,
                   celltype_col="cell_type",
                   normalize=normalize,
                   dr=dr,
                   zero_inflated=zero_inflated,
                   distribution_type=distribution_type) 
    cyto.import_data(cell_by_gene=cell_by_gene,
                     cell_metadata=cell_metadata)
    cyto.initialize_parameters()
    cyto.training_loop(n_epoches=1)
    x_samples, z_samples = cyto.infer(target_batch_index=target_batch_index,
                                        get_normal_component=get_normal_component)
    assert (x_samples.shape[0]==4000) and (z_samples.shape[1]==2)
    
    
