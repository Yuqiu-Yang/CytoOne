import os 
# Data manipulation
import scanpy as sc 
import numpy as np 
import pandas as pd 
# Torch 
import torch 
import torch.nn as nn 
# Typing and other info 
from typing import Optional, Tuple, Union
from time import perf_counter
from contextlib import contextmanager
import psutil


@contextmanager
def process_time_ram(message: str=""):
    """Monitor time taken as well as memory usage

    Parameters
    ----------
    message : str, optional
        The message to be printed, by default ""
    """
    print("="*30)
    print(message)
    t1 = t2 = perf_counter()
    yield lambda: t2-t1
    t2 = perf_counter()
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print("Done. Time taken is {0:.4f} seconds.".format(t2-t1))
    print("="*30)


def import_data(cell_by_gene: Union[str, pd.DataFrame],
                cell_metadata: Union[str, pd.DataFrame],
                batch_index_col: Optional[str]=None,
                celltype_col: Optional[str]=None,
                normalize: bool=True,
                dr: bool=False,
                zero_inflated: bool=True) -> sc.AnnData:
    """Read in cell-by-gene matrix and meta information as an anndata object 

    Parameters
    ----------
    cell_by_gene : Union[str, pd.DataFrame]
        Either the path the .csv file or a pandas dataframe. We assume that 
        the first column is the cell ID  
    cell_metadata : Union[str, pd.DataFrame]
        Either the path the .csv file or a pandas dataframe. We assume that 
        the first column is the cell ID 
    batch_index_col : Optional[str], optional
        The column name in cell_metadata that contains the batch information for each cell, by default None
    celltype_col : Optional[str], optional
        The column name in cell_metadata that contains the cell type information for each cell, by default None
    normalize : bool, optional
        Whether or not to perform arcsinh-transformation, by default True
    dr : bool, optional
        Whether or not to perfrom UMAP, by default False
    zero_inflated : bool, optional
        Whether or not the data is zero-inflated, by default True

    Returns
    -------
    sc.AnnData
        An AnnData object containing curated information 

    """
    with process_time_ram("Processing cell metadata") as ctm: 
        if isinstance(cell_metadata, str) and ("csv" in os.path.splitext(cell_metadata)[1]):
            cell_meta = pd.read_csv(cell_metadata, index_col=0)
        elif isinstance(cell_metadata, pd.DataFrame):
            cell_meta = cell_metadata.copy()
        else: 
            raise TypeError("Only .csv file or pandas dataframe is allowed")
        cell_meta.index.rename(name='cell_id', inplace=True)
    
    with process_time_ram("Processing cell-by-gene matrix") as ctm: 
        if isinstance(cell_by_gene, str) and ("csv" in os.path.splitext(cell_by_gene)[1]):
            counts = pd.read_csv(cell_by_gene, index_col=0)
        elif isinstance(cell_by_gene, pd.DataFrame):
            counts = cell_by_gene.copy()
        else: 
            raise TypeError("Only .csv file or pandas dataframe is allowed")
        
        counts.index.rename(name='cell_id', inplace=True)

    with process_time_ram("Creating AnnData object") as ctm:
        adata = sc.AnnData(counts)
        # Add cell type information 
        if celltype_col is not None:
            adata.obs['cell_type'] = cell_meta.loc[adata.obs_names, celltype_col]
        else: 
            adata.obs['cell_type'] = "cell"
        # Create index for each cell type 
        adata.obs['leiden'] = pd.factorize(adata.obs['cell_type'])[0].astype(str)
        # Add batch information 
        if batch_index_col is None:
            adata.obs['batch'] = "0"
        else:  
            adata.obs['batch'] = cell_meta.loc[adata.obs_names, batch_index_col]
        
        adata.obs['batch_index'] = pd.factorize(adata.obs['batch'])[0].astype(int)
        # Add basic info
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
        adata.uns['zero_inflated'] = zero_inflated
    # If normalization is needed 
    if normalize:
        with process_time_ram("ArcSinh transform") as ctm:
            if isinstance(adata.X, np.ndarray):
                adata.X = np.arcsinh(adata.X / 5.0)
            else:
                # For sparse matrices
                adata.X = adata.X.toarray()
                adata.X = np.arcsinh(adata.X / 5.0)
    # If zero inflated, we make sure the lower bound is 0
    if zero_inflated:
        with process_time_ram("Clipping the data") as ctm:
            adata.X = np.clip(adata.X, a_min=0, a_max=None)
    # UMAP
    if dr:
        with process_time_ram("UMAP") as ctm:
            sc.pp.pca(adata)
            sc.pp.neighbors(adata)    
            sc.tl.umap(adata)

    return adata 


def generate_strata(adata: sc.AnnData, 
                    n_strata: int=100) -> pd.DataFrame:
    """Generate strata of the counts for sampling 
    This ensures that all batches are included in each minibatch 

    Parameters
    ----------
    adata : sc.AnnData
        An AnnData object 
    n_strata : int, optional
        The number of strata, by default 100

    Returns
    -------
    pd.DataFrame
        A pd.DataFrame with batch information and strata information 
    """
    adata_w_batch = adata.to_df().copy()
    adata_w_batch['batch_index'] = adata.obs['batch_index'].copy()
    # Randomly suffle each batch 
    adata_w_batch = adata_w_batch.groupby(['batch_index'], observed=True, as_index=True).apply(lambda x: x.sample(frac=1, replace=False),
                                                            include_groups=False)
    adata_w_batch.reset_index(drop=False, names=['batch_index', "id_to_drop"], inplace=True)
    adata_w_batch.drop(columns=['id_to_drop'], inplace=True)
    # Split each batch into n_strata
    split_result = []
    for batch, group in adata_w_batch.groupby("batch_index"):
        splits = np.array_split(group.index, n_strata)
        for i, row_ind in enumerate(splits):
            part = group.loc[row_ind, :].copy()
            part["batch_index"] = batch
            part["stratum"] = i
            split_result.append(part)
    # Concatenate the results 
    adata_w_batch_strata = pd.concat(split_result).reset_index(drop=True)
    return adata_w_batch_strata


def load_stratum(adata_w_batch_strata: pd.DataFrame, 
                 stratum_id: int, 
                 target_batch_index: Optional[int]=None,
                 model_device='cpu') -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """_summary_

    Parameters
    ----------
    adata_w_batch_strata : pd.DataFrame
        _description_
    stratum_id : int
        _description_
    target_batch_index : Optional[int], optional
        _description_, by default None
    model_device : str, optional
        _description_, by default 'cpu'

    Returns
    -------
    Tuple[torch.tensor, torch.tensor, torch.tensor]
        _description_
    """
    cell_patch = adata_w_batch_strata.loc[adata_w_batch_strata['stratum'] == stratum_id, :].copy()
    source_batch_index = torch.tensor(cell_patch['batch_index'].values, dtype=torch.int64, device=model_device)
    if target_batch_index is None:
        target_batch_index = source_batch_index.clone()
    else:
        target_batch_index = torch.ones_like(source_batch_index)*target_batch_index

    cell_patch.drop(columns=['stratum', 'batch_index'], inplace=True)
    cell_by_gene_counts = torch.tensor(cell_patch.values, dtype=torch.float32, device=model_device)
    
    return cell_by_gene_counts, source_batch_index, target_batch_index

    
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, 
                 hidden_dims=[512, 256],
                 drop_out_p=0.2):
        super().__init__()

        self.skip_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        self.residual_seq = nn.ModuleList()
        current_d = in_dim
        for h_d in hidden_dims:
            self.residual_seq.append(nn.Sequential(
                nn.Linear(current_d, h_d),
                nn.LayerNorm(h_d),
                nn.GELU(),
                nn.Dropout(drop_out_p)
            ))
            current_d = h_d       
        self.residual_seq.append(nn.Sequential(
                    nn.Linear(current_d, out_dim),
                    nn.LayerNorm(out_dim)
                ))  
    def forward(self, x):
        skip_x = self.skip_proj(x)
        for s in self.residual_seq:
            x = s(x)
        return skip_x + 0.1*x


def kl_delta(delta_mu, delta_log_var, mu, log_var):
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)

    loss = -0.5 * torch.sum(1 + delta_log_var - delta_mu ** 2 / var - delta_var, dim=[1])
    return torch.mean(loss, dim=0)


def kl_standard(mu: torch.tensor,
                log_var: torch.tensor) -> torch.tensor:
    loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=[1])
    return torch.mean(loss, dim=0)


def reparameterize(mu: torch.tensor,
                   std: torch.tensor):
    z = torch.randn_like(mu) * std + mu
    return z


def compute_mmd(x, y):
    """
    Compute Maximum Mean Discrepancy between two distributions
    """
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    
    # RBF Kernel
    bandwidth_range = [0.01, 0.1, 1, 10]
    xx_sum = torch.diag(xx).unsqueeze(1).expand(xx.size(0), xx.size(0))
    xx_rbf = xx_sum + xx_sum.t() - 2*xx
    
    yy_sum = torch.diag(yy).unsqueeze(1).expand(yy.size(0), yy.size(0))
    yy_rbf = yy_sum + yy_sum.t() - 2*yy
    
    xy_sum_1 = torch.diag(xx).unsqueeze(1).expand(xx.size(0), yy.size(0))
    xy_sum_2 = torch.diag(yy).unsqueeze(0).expand(xx.size(0), yy.size(0))
    xy_rbf = xy_sum_1 + xy_sum_2 - 2*xy
    
    mmd = 0
    for bandwidth in bandwidth_range:
        xx_kernel = torch.exp(-xx_rbf / (2 * bandwidth ** 2))
        yy_kernel = torch.exp(-yy_rbf / (2 * bandwidth ** 2))
        xy_kernel = torch.exp(-xy_rbf / (2 * bandwidth ** 2))
        
        mmd += torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2 * torch.mean(xy_kernel)
            
    return mmd


def mmd_loss(z, batch_index):
    unique_batches = batch_index.unique()
    mmd_sum = 0
    count = 0
    
    for i in range(len(unique_batches)):
        for j in range(i + 1, len(unique_batches)):
            idx_i = (batch_index == unique_batches[i])
            idx_j = (batch_index == unique_batches[j])
            
            zi = z[idx_i]
            zj = z[idx_j]
            
            mmd = compute_mmd(x=zi,
                              y=zj)
            
            mmd_sum += mmd
            count += 1
    return mmd_sum / count if count > 0 else 0