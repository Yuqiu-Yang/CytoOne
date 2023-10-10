# Data IO 
import numpy as np 
import pandas as pd 

# PyTorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.distributions as d 
from torch.distributions.kl import register_kl
from pyro.distributions import Delta 
from torch.distributions import Normal, Categorical
from Cytomulate_Inv.basic_distributions import zero_inflated_lognormal

# Typing 
from typing import Optional, Tuple


@register_kl(Delta, Delta)
def _kl_delta_delta(p, q) -> torch.tensor:
    return -q.log_density 

@register_kl(Delta, zero_inflated_lognormal)
def _kl_delta_ziln(p, q) -> torch.tensor:
    return -q.log_prob(p.v) 

@register_kl(Delta, Categorical)
def _kl_delta_categorical(p, q) -> torch.tensor:
    return -q.log_prob(p.v) 

def data_curation(data_path: str, 
                  batch_index_col_name: Optional[str]=None,
                  condition_index_col_name: Optional[str]=None,
                  subject_index_col_name: Optional[str]=None,
                  other_nonprotein_col_names: Optional[list]=None,
                  arcsinh5_transform: bool=True,
                  **kwargs) -> Tuple[pd.DataFrame, dict]:
    assert isinstance(batch_index_col_name, str) or (batch_index_col_name is None), "batch_index_col_name should be a string or None"
    assert isinstance(condition_index_col_name, str) or (condition_index_col_name is None), "condition_index_col_name should be a string or None"
    assert isinstance(subject_index_col_name, str) or (subject_index_col_name is None), "subject_index_col_name should be a string or None"
    assert isinstance(other_nonprotein_col_names, list) or (other_nonprotein_col_names is None), "other_nonprotein_col_names should be a list of strings or None"
    
    df = pd.read_csv(data_path, 
                     **kwargs)
    nonprotein_col_names = []
    if batch_index_col_name is not None:
        batch_index, _ = pd.factorize(df[batch_index_col_name])
        nonprotein_col_names.append(batch_index_col_name)
    else:
        batch_index = np.zeros(df.shape[0], dtype=np.int32)
    if condition_index_col_name is not None:
        condition_index, _ = pd.factorize(df[condition_index_col_name])
        nonprotein_col_names.append(condition_index_col_name)
    else:
        condition_index = np.zeros(df.shape[0], dtype=np.int32)
    if subject_index_col_name is not None:
        subject_index, _ = pd.factorize(df[subject_index_col_name])
        nonprotein_col_names.append(subject_index_col_name)
    else:
        subject_index = np.zeros(df.shape[0], dtype=np.int32)
    
    if other_nonprotein_col_names is not None:
        nonprotein_col_names += other_nonprotein_col_names
    
    protein_col_names = list(df.columns[~df.columns.isin(nonprotein_col_names)])
    
    df.drop(columns=nonprotein_col_names, inplace=True)
    if arcsinh5_transform:
        df = np.arcsinh(df/5)
    
    df['batch_index'] = batch_index
    df['condition_index'] = condition_index
    df['subject_index'] = subject_index
    
    n_b = df['batch_index'].max(axis=0)+1
    n_c = df['condition_index'].max(axis=0)+1
    n_s = df['subject_index'].max(axis=0)+1
    
    meta_dict = {
        "batch_index_col_name": batch_index_col_name,
        "condition_index_col_name": condition_index_col_name,
        "subject_index_col_name": subject_index_col_name,
        "other_nonprotein_col_names": other_nonprotein_col_names,
        "protein_col_names": protein_col_names,
        "n_batches": n_b,
        "n_conditions": n_c,
        "n_subjects": n_s,
        "N": df.shape[0],
        "y_dim": len(protein_col_names)
    }
    
    return df, meta_dict
    
     

class cytof_dataset_class(Dataset):
    def __init__(self, 
                 df: pd.DataFrame,
                 meta_dict: dict) -> None:
        super().__init__()
        
        self.y = torch.tensor(df[meta_dict['protein_col_names']].values, dtype=torch.float32)
        
        b = torch.tensor(df['batch_index'].values, dtype=torch.int64).unsqueeze(1)
        c = torch.tensor(df['condition_index'].values, dtype=torch.int64).unsqueeze(1)
        s = torch.tensor(df['subject_index'].values, dtype=torch.int64).unsqueeze(1)
        
        self.FB = torch.zeros(meta_dict['N'], meta_dict['n_batches'])
        self.FC = torch.zeros(meta_dict['N'], meta_dict['n_conditions'])
        self.RS = torch.zeros(meta_dict['N'], meta_dict['n_subjects'])
        
        self.FB.scatter_(1, b, 1)
        self.FC.scatter_(1, c, 1)
        self.RS.scatter_(1, s, 1)
    
    def __len__(self) -> int:
        return self.y.shape[0]
    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        return self.y[index, :], self.FB[index, :], self.FC[index, :], self.RS[index, :]


def create_cytof_dataloader(df: pd.DataFrame,
                            meta_dict: dict,
                            batch_size: int,
                            shuffle: bool=True,
                            **kwargs) -> DataLoader:
    cytof_dataset = cytof_dataset_class(df=df,
                                        meta_dict=meta_dict,
                                        **kwargs)
    return DataLoader(cytof_dataset, batch_size=batch_size, shuffle=shuffle)

