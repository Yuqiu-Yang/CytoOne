# Data IO 
import os 
import numpy as np 
import pandas as pd 

# PyTorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Typing 
from typing import Optional


class dataset_class(Dataset):
    def __init__(self, 
                 df: pd.DataFrame,
                 meta_dict: Optional[dict]=None,
                 return_cell_types: bool=False) -> None:
        """Dataset class for all subsequent models 
        
        

        Parameters
        ----------
        df : pd.DataFrame
            A curated CyTOF dataframe  
        meta_dict : Optional[dict], optional
            _description_, by default None
        return_cell_types : bool, optional
            _description_, by default False
        """
        super().__init__()
        
        self.data_type = None
        
        if meta_dict is None:
            self.data_type = "pretrain"
            self.y = torch.tensor(df[['y']].values, dtype=torch.float32)
        else:
            if return_cell_types:
                self.data_type = "post training"
            else:
                self.data_type = "training"
            self.y = torch.tensor(df[meta_dict['protein_col_names']].values, dtype=torch.float32)
            
            b = torch.tensor(df['batch_index'].values, dtype=torch.int64).unsqueeze(1)
            c = torch.tensor(df['condition_index'].values, dtype=torch.int64).unsqueeze(1)
            s = torch.tensor(df['subject_index'].values, dtype=torch.int64).unsqueeze(1)
            self.type_index = torch.tensor(df['cell_type_index'].values, dtype=torch.int64).unsqueeze(1)
            
            self.FB = torch.zeros(meta_dict['N'], meta_dict['n_batches'])
            self.FC = torch.zeros(meta_dict['N'], meta_dict['n_conditions'])
            self.RS = torch.zeros(meta_dict['N'], meta_dict['n_subjects'])
            
            self.FB.scatter_(1, b, 1)
            self.FC.scatter_(1, c, 1)
            self.RS.scatter_(1, s, 1)
    
    def __len__(self) -> int:
        return self.y.shape[0]
    
    def __getitem__(self, index) -> dict:
        if self.data_type == "pretrain":
            return {"y": self.y[index, :]}
        elif self.data_type == "training":
            return {"y": self.y[index, :], 
                    "FB": self.FB[index, :],
                    "FC": self.FC[index, :], 
                    "RS": self.RS[index, :]}
        else:
            return {"y": self.y[index, :], 
                    "FB": self.FB[index, :],
                    "FC": self.FC[index, :], 
                    "RS": self.RS[index, :],
                    "types": self.type_index[index, :]}


def generate_dataloader(df: pd.DataFrame,
                        meta_dict: Optional[dict]=None,
                        batch_size: int=256,
                        shuffle: bool=True) -> DataLoader:
    
    dataset = dataset_class(df=df,
                            meta_dict=meta_dict)
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
                  save_every_epoch: bool=False,
                  print_every_n_minibatch: int=10,
                  cytoone_model=None) -> None:

    model._update_stage(stage=training_stage)
    
    best_val_loss = np.inf
    for epoch in range(1, n_epoches + 1):
        # First train on 
        model.train()
        for minibatch, data in enumerate(train_dataloader):
            if training_stage in ["pretrain", "dimension reduction", "clustering"]:
                distribution_dict = model(**data) 
            else:
                (_, _, q_pi_dict, z_w_samples, _) = cytoone_model.get_posterior_samples(**data)
                one_hot_index = q_pi_dict['one_hot_encoding']
                z = z_w_samples['z']
                w = z_w_samples['w']
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
            
            if (epoch % 1 == 0) and (minibatch % print_every_n_minibatch == 0):
                print("Epoch {}, minibatch {}. Training loss is {}\n".format(epoch, minibatch, training_loss.item()))
                print("="*25)
                
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for data in val_dataloader:
                if training_stage in ["pretrain", "dimension reduction", "clustering"]:
                    val_distributions_dict = model(**data) 
                else:
                    (_, _, q_pi_dict, z_w_samples, _) = cytoone_model.get_posterior_samples(**data)
                    one_hot_index = q_pi_dict['one_hot_encoding']
                    z = z_w_samples['z']
                    w = z_w_samples['w']
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
                }, os.path.join(models_dir, model_name + "_" + str(epoch) + ".pth"))
            else:
                if (total_val_loss/len(val_dataloader)) < best_val_loss:
                    best_val_loss = validation_loss.item()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(models_dir, model_name + ".pth"))
        
        if scheduler is not None:
            scheduler.step()
        
        if epoch % 1 == 0:
            print('Epoch {} Validation loss {}\n'.format(epoch, total_val_loss/len(val_dataloader)))
            print("="*25) 

