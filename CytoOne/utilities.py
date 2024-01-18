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
    
    
def generate_cytoone_model_input(meta_dict: dict,
                                 x_dim: int, 
                                 n_cell_types: int,
                                 model_check_point_path: Optional[str]=None):
    
    parameter_dict = {
        "y_dim": meta_dict['y_dim'],
        "x_dim": x_dim,
        "n_batches": meta_dict['n_batches'],
        "n_conditions": meta_dict['n_conditions'],
        "n_subjects": meta_dict['n_subjects'],
        "n_cell_types": n_cell_types,
        "model_check_point_path": model_check_point_path
    }
    
    return parameter_dict


def generate_pretrain_data(y_scale: float=0.1,
                           n_sample: int=100000):
    z = np.random.uniform(low=-5, high=5, size=n_sample)
    w = np.random.uniform(low=-5, high=5, size=n_sample)
    pure_y = np.round(1/(1+np.exp(-w))) * np.exp(z)
    y = pure_y + np.random.normal(loc=0, scale=y_scale, size=n_sample)
    df = pd.DataFrame({"y": y,
                       "z": z, 
                       "w": w})
    return df


class pretrain_dataset_class(Dataset):
    def __init__(self,
                 df: pd.DataFrame) -> None:
        super().__init__()
        self.y = torch.tensor(df[['y']].values, dtype=torch.float32)
    
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return {"y": self.y[index,:]}


class cytoone_dataset_class(Dataset):
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
    def __getitem__(self, index) -> dict:
        return {"y": self.y[index, :], 
                "FB": self.FB[index, :],
                "FC": self.FC[index, :], 
                "RS": self.RS[index, :]}


def generate_dataloader(df: pd.DataFrame,
                        meta_dict: Optional[dict]=None,
                        batch_size: int=256,
                        shuffle: bool=True,
                        **kwargs) -> DataLoader:
    if meta_dict is None:
        dataset = pretrain_dataset_class(df=df)
    else:
        dataset = cytoone_dataset_class(df=df,
                                        meta_dict=meta_dict,
                                        **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def training_loop(model,
                  optimizer,
                  scheduler, 
                  n_epoches: int,
                  train_dataloader: DataLoader,
                  val_dataloader: DataLoader,
                  models_dir: str,
                  model_name: str,
                  training_stage: str,
                  save_every_epoch: bool=True,
                  cytoone_model=None) -> None:

    model._update_stage(stage=training_stage)
    
    best_val_loss = -np.inf
    for epoch in range(1, n_epoches + 1):
        # First train on 
        model.train()
        for minibatch, data in enumerate(train_dataloader):
            if training_stage in ["pretrain", "dimension reduction", "clustering"]:
                distribution_dict = model(**data) 
            else:
                (z, w, _, _, one_hot_index) = cytoone_model.get_posterior_samples(**data)
                if training_stage == "abundance effect estimation":
                    distribution_dict = model(one_hot_index=one_hot_index,
                                              FC=data["FC"],
                                              RS=data["RS"]) 
                elif training_stage == "expression effect estimation":
                    if model.effect_type == "expression":
                        distribution_dict = model(z_w=z,
                                                  one_hot_index=one_hot_index,
                                                  FB=data['FB'],
                                                  FC=data['FC'],
                                                  RS=data['RS']) 
                    elif model.effect_type == "inflation":
                        distribution_dict = model(z_w=w,
                                                  one_hot_index=one_hot_index,
                                                  FB=data['FB'],
                                                  FC=data['FC'],
                                                  RS=data['RS'])  
                    else: 
                        raise RuntimeError("Illegal training stage")  
                else: 
                    raise RuntimeError("Illegal training stage")  
                
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
            for data in val_dataloader:
                if training_stage in ["pretrain", "dimension reduction", "clustering"]:
                    val_distributions_dict = model(**data) 
                else:
                    (z, w, _, _, one_hot_index) = cytoone_model.get_posterior_samples(**data)
                    if training_stage == "abundance effect estimation":
                        val_distributions_dict = model(one_hot_index=one_hot_index,
                                                       FC=data['FC'],
                                                       RS=data['RS']) 
                    elif training_stage == "expression effect estimation":
                        if model.effect_type == "expression":
                            val_distributions_dict = model(z_w=z,
                                                           one_hot_index=one_hot_index,
                                                           FB=data['FB'],
                                                           FC=data['FC'],
                                                           RS=data['RS']) 
                        elif model.effect_type == "inflation":
                            val_distributions_dict = model(z_w=w,
                                                           one_hot_index=one_hot_index,
                                                           FB=data['FB'],
                                                           FC=data['FC'],
                                                           RS=data['RS'])  
                        else: 
                            raise RuntimeError("Illegal training stage") 
                    else: 
                        raise RuntimeError("Illegal training stage")  
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



# class abundance_dataset_class(Dataset):
#     def __init__(self, 
#                  one_hot_index: torch.tensor,
#                  FC: torch.tensor) -> None:
#         super().__init__()
        
#         self.one_hot_index = one_hot_index
        
#         self.FC = FC
 
#     def __len__(self) -> int:
#         return self.one_hot_index.shape[0]
#     def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
#         return self.one_hot_index[index, :], self.FC[index, :]


# def create_abundance_dataloader(one_hot_index: torch.tensor,
#                                 FC: torch.tensor,
#                                 batch_size: int,
#                                 shuffle: bool=True) -> DataLoader:
#     abundance_dataset = abundance_dataset_class(one_hot_index=one_hot_index,
#                                                 FC=FC)
#     return DataLoader(abundance_dataset, batch_size=batch_size, shuffle=shuffle)


# def abundance_training_loop(model,
#                             optimizer,
#                             scheduler, 
#                             n_epoches,
#                             train_dataloader,
#                             val_dataloader,
#                             models_dir,
#                             model_name,
#                             save_every_epoch: bool=True) -> None:

#     best_val_loss = -np.inf
#     for epoch in range(1, n_epoches + 1):
#         # First train on 
#         model.train()
#         for minibatch, (y, FC) in enumerate(train_dataloader):
#             distribution_dict = model(one_hot_index=y,
#                                       FC=FC)
#             training_loss = model.compute_loss(distribution_dict=distribution_dict)

#             optimizer.zero_grad()
#             training_loss.backward()
#             #Gradient Value Clipping
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
#             optimizer.step()
            
#             if (epoch % 1 == 0) and (minibatch % 10 == 0):
#                 print("Epoch {}, minibatch {}. Training loss is {}\n".format(epoch, minibatch, training_loss.item()))
#                 print("="*25)
                
#         model.eval()
#         total_val_loss = 0.0
#         with torch.no_grad():
#             for (y, FC) in val_dataloader:
#                 val_distributions_dict = model(one_hot_index=y,
#                                                FC=FC)
#                 validation_loss = model.compute_loss(distribution_dict=val_distributions_dict)
#                 total_val_loss += validation_loss.item()
            
#             if save_every_epoch:
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict()
#                 }, os.path.join(models_dir + model_name + "_" + str(epoch) + ".pth"))
#             else:
#                 if (total_val_loss/len(val_dataloader)) < best_val_loss:
#                     best_val_loss = validation_loss.item()
#                     torch.save({
#                         'epoch': epoch,
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict()
#                     }, os.path.join(models_dir + model_name + ".pth"))
        
#         if scheduler is not None:
#             scheduler.step()
        
#         if epoch % 1 == 0:
#             print('Epoch {} Validation loss {}\n'.format(epoch, total_val_loss/len(val_dataloader)))
#             print("="*25) 


# class expression_dataset_class(Dataset):
#     def __init__(self, 
#                  z: torch.tensor,
#                  one_hot_index: torch.tensor,
#                  FB: torch.tensor,
#                  FC: torch.tensor) -> None:
#         super().__init__()
        
#         self.z = z
#         self.one_hot_index = one_hot_index
#         self.FB = FB
#         self.FC = FC
 
#     def __len__(self) -> int:
#         return self.one_hot_index.shape[0]
#     def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
#         return self.z[index, :], self.one_hot_index[index, :], self.FB[index, :], self.FC[index, :]


# def create_expression_dataloader(z: torch.tensor,
#                                 one_hot_index: torch.tensor,
#                                 FB: torch.tensor,
#                                 FC: torch.tensor,
#                                 batch_size: int,
#                                 shuffle: bool=True) -> DataLoader:
#     expression_dataset = expression_dataset_class(z=z,
#                                                 one_hot_index=one_hot_index,
#                                                 FB=FB,
#                                                 FC=FC)
#     return DataLoader(expression_dataset, batch_size=batch_size, shuffle=shuffle)


# def expression_training_loop(model,
#                             optimizer,
#                             scheduler, 
#                             n_epoches,
#                             train_dataloader,
#                             val_dataloader,
#                             models_dir,
#                             model_name,
#                             save_every_epoch: bool=True) -> None:

#     best_val_loss = -np.inf
#     for epoch in range(1, n_epoches + 1):
#         # First train on 
#         model.train()
#         for minibatch, (z, one_hot_index, FB, FC) in enumerate(train_dataloader):
#             distribution_dict = model(z=z,
#                                       one_hot_index=one_hot_index,
#                                       FB=FB,
#                                       FC=FC)
#             training_loss = model.compute_loss(distribution_dict=distribution_dict)

#             optimizer.zero_grad()
#             training_loss.backward()
#             #Gradient Value Clipping
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
#             optimizer.step()
            
#             if (epoch % 1 == 0) and (minibatch % 10 == 0):
#                 print("Epoch {}, minibatch {}. Training loss is {}\n".format(epoch, minibatch, training_loss.item()))
#                 print("="*25)
                
#         model.eval()
#         total_val_loss = 0.0
#         with torch.no_grad():
#             for (z, one_hot_index, FB, FC) in val_dataloader:
#                 val_distributions_dict = model(z=z,
#                                                one_hot_index=one_hot_index,
#                                                FB=FB,
#                                                FC=FC)
#                 validation_loss = model.compute_loss(distribution_dict=val_distributions_dict)
#                 total_val_loss += validation_loss.item()
            
#             if save_every_epoch:
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict()
#                 }, os.path.join(models_dir + model_name + "_" + str(epoch) + ".pth"))
#             else:
#                 if (total_val_loss/len(val_dataloader)) < best_val_loss:
#                     best_val_loss = validation_loss.item()
#                     torch.save({
#                         'epoch': epoch,
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict()
#                     }, os.path.join(models_dir + model_name + ".pth"))
        
#         if scheduler is not None:
#             scheduler.step()
        
#         if epoch % 1 == 0:
#             print('Epoch {} Validation loss {}\n'.format(epoch, total_val_loss/len(val_dataloader)))
#             print("="*25) 
