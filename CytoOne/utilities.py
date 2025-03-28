
import os 

import scanpy as sc 
import numpy as np 
import pandas as pd 

import torch 
# Typing 
from typing import Optional, Tuple, Union


def import_data(cell_by_gene: Union[str, pd.DataFrame],
                cell_metadata: Union[str, pd.DataFrame],
                batch_index_col: Optional[str]=None,
                celltype_col: Optional[str]=None,
                normalize: bool=True,
                dr: bool=False):
    
    if isinstance(cell_metadata, str) and ("csv" in os.path.splitext(cell_metadata)[1]):
        cell_meta = pd.read_csv(cell_metadata, index_col=0)
    elif isinstance(cell_metadata, pd.DataFrame):
        cell_meta = cell_metadata.copy()
    else: 
        raise TypeError("Only .csv file or pandas dataframe is allowed")
    cell_meta.index.rename(name='cell_id', inplace=True)
    
    if isinstance(cell_by_gene, str) and ("csv" in os.path.splitext(cell_by_gene)[1]):
        counts = pd.read_csv(cell_by_gene, index_col=0)
    elif isinstance(cell_by_gene, pd.DataFrame):
        counts = cell_by_gene.copy()
    else: 
        raise TypeError("Only .csv file or pandas dataframe is allowed")
    
    counts.index.rename(name='cell_id', inplace=True)

    adata = sc.AnnData(counts)
    if celltype_col is not None:
        adata.obs['cell_type'] = cell_meta.loc[adata.obs_names, celltype_col]
    else: 
        adata.obs['cell_type'] = "cell"

    adata.obs['leiden'] = pd.factorize(adata.obs['cell_type'])[0].astype(str)

    if batch_index_col is None:
        adata.obs['batch'] = "0"
    else:  
        adata.obs['batch'] = cell_meta.loc[adata.obs_names, batch_index_col]
    
    adata.obs['batch_index'] = pd.factorize(adata.obs['batch'])[0].astype(int)

    adata.uns['unique_leiden'] = np.unique(adata.obs['leiden'])
    adata.uns['n_leiden'] = len(adata.uns['unique_leiden'])
    adata.uns['cell_type_leiden_map'] = adata.obs[["cell_type", "leiden"]].drop_duplicates(ignore_index=True)
    adata.uns['cell_type_leiden_map'].rename(columns={"cell_type": "cell_type_name",
                                                    "leiden": "cell_type_index"},
                                            inplace=True)
    
    adata.uns['unique_batch'] = np.unique(adata.obs['batch_index'])
    adata.uns['n_batches'] = len(adata.uns['unique_batch'])
    adata.uns['batch_index_batch_map'] = adata.obs[["batch", "batch_index"]].drop_duplicates(ignore_index=True)

    adata.uns['n_genes'] = adata.var.shape[0]

    if normalize:
        if isinstance(adata.X, np.ndarray):
            adata.X = np.arcsinh(adata.X / 5.0)
        else:
            # For sparse matrices
            adata.X = adata.X.toarray()
            adata.X = np.arcsinh(adata.X / 5.0)
    
    if dr:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)    
        sc.tl.umap(adata)

    return adata 


def generate_strata(adata, 
                    n_splits: int=100,
                    batch_index: Optional[int]=None):
    adata_w_batch = adata.to_df().copy()
    adata_w_batch['batch_index'] = adata.obs['batch_index'].copy()
    if batch_index is not None:
        adata_w_batch = adata_w_batch.loc[adata_w_batch['batch_index']==batch_index, :]

    adata_w_batch = adata_w_batch.groupby(['batch_index'], observed=True, as_index=True).apply(lambda x: x.sample(frac=1, replace=False),
                                                            include_groups=False)
    adata_w_batch.reset_index(drop=False, names=['batch_index', "id_to_drop"], inplace=True)
    adata_w_batch.drop(columns=['id_to_drop'], inplace=True)

    split_result = []

    for batch, group in adata_w_batch.groupby("batch_index"):
        splits = np.array_split(group, n_splits)
        for i, part in enumerate(splits):
            part = part.copy()
            part["batch_index"] = batch
            part["stratum"] = i
            split_result.append(part)

    adata_w_batch_strata = pd.concat(split_result).reset_index(drop=True)
    return adata_w_batch_strata


def load_stratum(adata_w_batch_strata, 
                 stratum_id, 
                 model_device='cpu'):
    cell_patch = adata_w_batch_strata.loc[adata_w_batch_strata['stratum'] == stratum_id, :]
    source_batch_index = torch.tensor(cell_patch['batch_index'], dtype=torch.int64, device=model_device)
    target_batch_index = torch.tensor(np.random.permutation(cell_patch['batch_index'].values), dtype=torch.int64, device=model_device)

    cell_patch.drop(columns=['stratum', 'batch_index'], inplace=True)
    cell_by_gene_counts = torch.tensor(cell_patch.values, dtype=torch.float32, device=model_device)
    
    return cell_by_gene_counts, source_batch_index, target_batch_index

    


