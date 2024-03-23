# Data IO
import pandas as pd 
import numpy as np 

# PyTorch
import torch 
from torch.utils.data import Dataset, DataLoader


class cytof_dataset_class(Dataset):
    def __init__(self, 
                 df: pd.DataFrame,
                 protein_col_names: list):
        super().__init__()
        
        self.y = torch.tensor(df[protein_col_names], dtype=torch.float32)
        
        n_b = df['batch_index'].max(axis=0)+1
        n_c = df['condition_index'].max(axis=0)+1
        n_s = df['subject_index'].max(axis=0)+1
        N = df.shape[0]
        b = torch.tensor(df['batch_index'], dtype=torch.int32)
        c = torch.tensor(df['condition_index'], dtype=torch.int32)
        s = torch.tensor(df['subject_index'], dtype=torch.int32)
        
        self.FB = torch.zeros(N, n_b)
        self.FC = torch.zeros(N, n_c)
        self.RS = torch.zeros(N, n_s)
        
        self.FB.scatter_(1, b, 1)
        self.FC.scatter_(1, c, 1)
        self.RS.scatter_(1, s, 1)
    
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, index):
        return self.y[index, :], self.FB[index, :], self.FC[index, :], self.RS[index, :]


def create_cytof_dataloader(df: pd.DataFrame,
                            protein_col_names: list,
                            batch_size: int,
                            shuffle: bool,
                            **kwargs):
    cytof_dataset = cytof_dataset_class(df=df,
                                        protein_col_names=protein_col_names,
                                        **kwargs)
    return DataLoader(cytof_dataset, batch_size=batch_size, shuffle=shuffle)

