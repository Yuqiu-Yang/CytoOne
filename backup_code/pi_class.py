import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Categorical, Independent
from pyro.distributions import Delta 

from CytoOne.base_class import component_base_class

import numpy as np 
from collections import OrderedDict


class p_effect_pi_class(component_base_class):
    def __init__(self,
                 model_device: torch.device):
        super().__init__(stage_to_change="abundance effect estimation",
                         distribution_info={"pi": None})
        self.model_device = model_device
        
    def _update_distributions(self, 
                              FC: torch.tensor,
                              RS: torch.tensor, 
                              beta_sample: torch.tensor,
                              gamma_sample: torch.tensor):
        
        logits = torch.matmul(FC, beta_sample) + torch.matmul(RS, gamma_sample)
        probs = F.softmax(logits, dim=1)
        self.distribution_dict['pi'] = Independent(Categorical(probs=probs),
                                                   reinterpreted_batch_ndims=0)


class p_pi_class(component_base_class):
    def __init__(self, 
                 model_device: torch.device,
                 n_cell_types: int,
                 n_conditions: int=1,
                 n_subjects: int=1) -> None:
        super().__init__(stage_to_change="clustering", 
                         distribution_info={"pi": None})
        self.model_device = model_device
        extra_n_dim = np.sum([n for n in [n_conditions, n_subjects] if n>1], dtype=int)
        if extra_n_dim > 1:
            self.in_dim = extra_n_dim
        else:
            self.in_dim = 1
        self.logit_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=self.in_dim,
                              out_features=n_cell_types,
                              bias=False))
        ]))
        
    def _update_distributions(self, 
                              FC: torch.tensor,
                              RS: torch.tensor):
        if self.in_dim == 1:
            effects = torch.ones((FC.shape[0], 1),
                                 device=self.model_device) 
        else:
            extra_effect_list = [m for m in [FC, RS] if m.shape[1] > 1]
            effects = torch.cat(extra_effect_list, dim=1)
            
        logits = self.logit_mapping(effects) 
        probs = F.softmax(logits, dim=1)
        self.distribution_dict['pi'] = Independent(Categorical(probs=probs),
                                                   reinterpreted_batch_ndims=0)    
        

class q_pi_class(component_base_class):
    def __init__(self,
                 model_device: torch.device,
                 x_dim: int,
                 n_cell_types: int,
                 vq_vae_weight: float) -> None:
        super().__init__(stage_to_change="clustering", 
                         distribution_info={"pi": None,
                                            "embedding": None,
                                            "vq_loss": None})
        self.model_device = model_device
        self.vq_vae_weight = vq_vae_weight
        self.cell_embeddings = nn.Embedding(num_embeddings=n_cell_types,
                                            embedding_dim=x_dim)
                    
    def _update_distributions(self, x):
        # x**2 should be an N * x_dim tensor
        # After torch.sum with keepdim=True,
        # the resulting tensor should be N * 1
        
        # The weight of cell_embeddings should be a 
        # P * x_dim tensor. After torch.sum with keepdim=False 
        # the resulting tensor should be P
        
        # Summing these two tensors up would by 
        # broadcasting give us an N * P tensor 
        
        distance = torch.sum(x**2, dim=1, keepdim=True) + \
                        torch.sum(self.cell_embeddings.weight**2, dim=1, keepdim=False) - \
                        2 * torch.matmul(x, self.cell_embeddings.weight.t())
        
        # Now for each row of X, we will find the index 
        # of the closest embedding 
        index = torch.argmin(distance, dim=1).unsqueeze(dim=1)
        
        # Then we convert this index to its one-hot encoding 
        one_hot_encoding = torch.zeros(distance.size(0), 
                                       distance.size(1))
        one_hot_encoding.scatter_(1, index, 1)
        
        quantized_latents = torch.matmul(one_hot_encoding, self.cell_embeddings.weight)
        
        # Compute VQ loss 
        commitment_loss = F.mse_loss(quantized_latents.detach(), x)
        embedding_loss = F.mse_loss(quantized_latents, x.detach())
        
        vq_loss = commitment_loss * self.vq_vae_weight  + embedding_loss
        
        quantized_latents = x + (quantized_latents-x).detach() 
        
        self.distribution_dict['pi'] = Independent(Delta(v=index.squeeze(1),
                                                         log_density=1), 
                                                         reinterpreted_batch_ndims=0)
        self.distribution_dict['embedding'] = quantized_latents
        self.distribution_dict['vq_loss'] = vq_loss
        self.distribution_dict['one_hot_encoding'] = one_hot_encoding
