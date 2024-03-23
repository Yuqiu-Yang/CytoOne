# Data IO 
import os 
from copy import deepcopy
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
from CytoOne.basic_distributions import zero_inflated_lognormal

# Typing 
from typing import Optional, Tuple, Union


################################
# New KL divergences 
# The formula is KL(p||q) = -\int p \log \dfrac{q}{p} 

@register_kl(Delta, Delta)
def _kl_delta_delta(p, q) -> torch.tensor:
    return -q.log_density 

@register_kl(Delta, zero_inflated_lognormal)
def _kl_delta_ziln(p, q) -> torch.tensor:
    return -q.log_prob(p.v) 

@register_kl(Delta, Categorical)
def _kl_delta_categorical(p, q) -> torch.tensor:
    return -q.log_prob(p.v) 


############################

def data_curation(user_data: Union[pd.DataFrame, str], 
                  batch_index_col_name: Optional[str]=None,
                  condition_index_col_name: Optional[str]=None,
                  subject_index_col_name: Optional[str]=None,
                  other_nonprotein_col_names: Optional[list]=None,
                  batches_to_retain: Optional[list]=None,
                  conditions_to_retain: Optional[list]=None,
                  subjects_to_retain: Optional[list]=None,
                  is_zero_inflated: bool=True,
                  arcsinh5_transform: bool=True,
                  **kwargs) -> Tuple[pd.DataFrame, dict]:
    assert isinstance(batch_index_col_name, str) or (batch_index_col_name is None), "batch_index_col_name should be a string or None"
    assert isinstance(condition_index_col_name, str) or (condition_index_col_name is None), "condition_index_col_name should be a string or None"
    assert isinstance(subject_index_col_name, str) or (subject_index_col_name is None), "subject_index_col_name should be a string or None"
    assert isinstance(other_nonprotein_col_names, list) or (other_nonprotein_col_names is None), "other_nonprotein_col_names should be a list of strings or None"
    
    if isinstance(user_data, str):
        df = pd.read_csv(user_data, **kwargs)
    else:
        df = deepcopy(user_data)
    if batches_to_retain is not None:
        df = df[df[batch_index_col_name].isin(batches_to_retain)].reset_index(drop=True)
    if conditions_to_retain is not None:
        df = df[df[condition_index_col_name].isin(conditions_to_retain)].reset_index(drop=True)
    if subjects_to_retain is not None:
        df = df[df[subject_index_col_name].isin(subjects_to_retain)].reset_index(drop=True)
    
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
    
    if is_zero_inflated:
        df = np.clip(df, a_min=0, a_max=None)
    
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
    
def generate_cytof_model_input(meta_dict: dict,
                               x_dim: int, 
                               distribution_type: str,
                               fixed_noise_level: bool,
                               n_cell_types: int,
                               delta: float,
                               sigma_y: float,
                               sigma_x: float,
                               sigma_alpha: float,
                               sigma_beta: float,
                               mu_log_var: float,
                               sigma_log_var: float,
                               vq_vae_weight: float,
                               global_weight: float,
                               local_weight: float):
    
    parameter_dict = {
        "y_dim": meta_dict['y_dim'],
        "x_dim": x_dim,
        "distribution_type": distribution_type,
        "fixed_noise_level": fixed_noise_level,
        "n_batches": meta_dict['n_batches'],
        "n_conditions": meta_dict['n_conditions'],
        "n_subjects": meta_dict['n_subjects'],
        "n_cell_types": n_cell_types,
        "sigma_y_dict": {"delta": delta,
                         "y": sigma_y},
        "sigma_x_dict": {"x": sigma_x},
        "sigma_alpha_dict": {
            "z_mu": sigma_alpha,
            "z_Sigma": sigma_alpha,
            "w_mu": sigma_alpha 
        },
        "sigma_beta_dict": {
            "pi": sigma_beta,
            "z_mu": sigma_beta,
            "z_Sigma": sigma_beta,
            "w_mu": sigma_beta
        },
        "mu_log_var_dict": {
            "pi": mu_log_var,
            "z_mu": mu_log_var,
            "z_Sigma": mu_log_var,
            "w_mu": mu_log_var
        },
        "sigma_log_var_dict": {
            "pi": sigma_log_var,
            "z_mu": sigma_log_var,
            "z_Sigma": sigma_log_var,
            "w_mu": sigma_log_var
        },
        "vq_vae_weight": vq_vae_weight,
        "global_weight": global_weight,
        "local_weight": local_weight
    }
    
    return parameter_dict



# class cytof_dataset_class:
#     def __init__(self, 
#                  df: pd.DataFrame,
#                  meta_dict: dict) -> None:
        
#         self.y = torch.tensor(df[meta_dict['protein_col_names']].values, dtype=torch.float32)
        
#         b = torch.tensor(df['batch_index'].values, dtype=torch.int64).unsqueeze(1)
#         c = torch.tensor(df['condition_index'].values, dtype=torch.int64).unsqueeze(1)
#         s = torch.tensor(df['subject_index'].values, dtype=torch.int64).unsqueeze(1)
        
#         self.FB = torch.zeros(meta_dict['N'], meta_dict['n_batches'])
#         self.FC = torch.zeros(meta_dict['N'], meta_dict['n_conditions'])
#         self.RS = torch.zeros(meta_dict['N'], meta_dict['n_subjects'])
        
#         self.FB.scatter_(1, b, 1)
#         self.FC.scatter_(1, c, 1)
#         self.RS.scatter_(1, s, 1)
        
        
        
        
        
#         # Check number of observations 
#         with open(self.data_path, mode='r') as f:
#             for count, _ in enumerate(f):
#                 pass 
#         self.data_len = load_size
#         # 
#         if load_size > count:
#             print("Load size greater than the dataset. Setting the load size to the size of the dataset\n")
#             self.data_len = count
#             self.tcrs = pd.read_csv(self.data_path,\
#                                     sep="\t",\
#                                     header=0)
#         else:
#             # 
#             skip_idx = np.random.choice(count,\
#                                         size=count - load_size,\
#                                         replace=False)
#             skip_idx += 1
#             self.tcrs = pd.read_csv(self.data_path,\
#                                     sep="\t",\
#                                     header=0,\
#                                     skiprows=skip_idx)

#     def __len__(self):
#         return self.data_len
    
#     def __getitem__(self, idx: list):
#         return self.tcrs.iloc[idx,:]
        

# class background_tcr_dataloader_class:
#     def __init__(self, 
#                  cytof_dataset,
#                  batch_size: Union[int, list]=10, 
#                  replacement: bool=True) -> None:
        
#         self.background_tcr_dataset = background_tcr_dataset
#         self.batch_sizes = batch_size
#         if isinstance(batch_size, int):
#             self.batch_size = batch_size    
#         else:
#             self.batch_size = 0
#         self.replacement = replacement
        
#     def __iter__(self):
#         return background_tcr_dataset_iterator(self)


# class background_tcr_dataset_iterator:
#     def __init__(self, background_tcr_dataloader: background_tcr_dataloader_class) -> None:
#         """Iterator class for a tcr dataset

#         Parameters
#         ----------
#         background_tcr_dataloader: background_tcr_dataloader_class
#             Dataloader for the dataset

#         """
#         self.background_tcr_dataloader = background_tcr_dataloader
#         self.batch_size_ind = -1

#     def __next__(self):
#         if isinstance(self.background_tcr_dataloader.batch_sizes, list):
#             self.batch_size_ind += 1
#             if self.batch_size_ind >= len(self.background_tcr_dataloader.batch_sizes):
#                 raise StopIteration
#             self.background_tcr_dataloader.batch_size = self.background_tcr_dataloader.batch_sizes[self.batch_size_ind]
        
#         idx = np.random.choice(len(self.background_tcr_dataloader.background_tcr_dataset),\
#                                size=self.background_tcr_dataloader.batch_size,\
#                                replace=self.background_tcr_dataloader.replacement)
#         return self.background_tcr_dataloader.background_tcr_dataset[idx.tolist()]



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



def training_loop(model,
                  optimizer,
                  scheduler, 
                  n_epoches,
                  train_dataloader,
                  val_dataloader,
                  models_dir,
                  model_name,
                  sigma_y: Optional[Union[np.ndarray, float]]=None,
                  save_every_epoch: bool=True) -> None:
    
    if sigma_y is None:
        sigma_y = [None] * n_epoches
    if isinstance(sigma_y, float):
        sigma_y = np.ones(n_epoches) * sigma_y
    
    
    best_val_loss = -np.inf
    for epoch in range(1, n_epoches + 1):
        # First train on 
        model.train()
        for minibatch, (y, FB, FC, RS) in enumerate(train_dataloader):
            distribution_dict = model(y=y,
                                      FB=FB,
                                      FC=FC,
                                      RS=RS,
                                      sigma_y=sigma_y[np.min([epoch-1, len(sigma_y)-1])])
            training_loss = model.compute_loss(distribution_dict=distribution_dict)

            optimizer.zero_grad()
            training_loss.backward()
            #Gradient Value Clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
            
            if (epoch % 1 == 0) and (minibatch % 10 == 0):
                print("Epoch {}, minibatch {}. Training loss is {}\n".format(epoch, minibatch, training_loss.item()))
                print("="*25)
                
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for (y, FB, FC, RS) in val_dataloader:
                val_distributions_dict = model(y=y,
                                               FB=FB,
                                               FC=FC,
                                               RS=RS,
                                               sigma_y=sigma_y[np.min([epoch-1, len(sigma_y)-1])])
                validation_loss = model.compute_loss(distribution_dict=val_distributions_dict)
                total_val_loss += validation_loss.item()
            
            if save_every_epoch:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(models_dir + model_name + "_" + str(epoch) + ".pth"))
            else:
                if (total_val_loss/len(val_dataloader)) < best_val_loss:
                    best_val_loss = validation_loss.item()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(models_dir + model_name + ".pth"))
        
        if scheduler is not None:
            scheduler.step()
        
        if epoch % 1 == 0:
            print('Epoch {} Validation loss {}\n'.format(epoch, total_val_loss/len(val_dataloader)))
            print("="*25) 




