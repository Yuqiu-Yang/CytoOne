# Data IO
import pandas as pd 
import numpy as np 

# PyTorch
import torch 
from torch.utils.data import Dataset, DataLoader


class cytof_dataset_class(Dataset):
    def __init__(self, data_path, 
                 batch_column, **kwargs):
        super().__init__()
        temp = pd.read_csv(data_path, **kwargs)
        if batch_column is None:
            self.y = torch.tensor(temp.values, dtype=torch.float32)
            self.b = torch.zeros((temp.shape[0],1))
        else:
            self.y = torch.tensor(temp.loc[:, temp.columns!=batch_column].values, dtype=torch.float32)
            self.b = torch.tensor(temp[batch_column].values, dtype=torch.float32)
    
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, index):
        return self.y[index, :], self.b[index, :]
            

def create_cytof_dataloader(data_path, 
                            batch_column,
                            batch_size,
                            shuffle,
                            **kwargs):
    cytof_dataset = cytof_dataset_class(data_path=data_path,
                                        batch_column=batch_column,
                                        **kwargs)
    return DataLoader(cytof_dataset, batch_size=batch_size, shuffle=shuffle)

