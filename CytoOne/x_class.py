import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from CytoOne.base_class import component_base_class

import numpy as np 
from collections import OrderedDict

from typing import Union, Optional


class p_x_class(component_base_class):
    def __init__(self, 
                 x_dim: int) -> None:
        self.x_dim = x_dim
        super().__init__(stage_to_change="dimension reduction",
                         distribution_info={"x": None})
        
    def _update_distributions(self,
                              embedding: Optional[torch.tensor]=None):
        if embedding is None:
            self.distribution_dict["x"] = Independent(Normal(loc=torch.zeros(self.x_dim),
                                                             scale=torch.ones(self.x_dim)), 
                                                        reinterpreted_batch_ndims=1) 
        else: 
            self.distribution_dict["x"] = Independent(Normal(loc=embedding,
                                                              scale=torch.ones(self.x_dim)), 
                                                        reinterpreted_batch_ndims=1) 
        
    
        
class q_x_class(component_base_class):
    def __init__(self,
                 y_dim: int,
                 x_dim: int,
                 n_batches: int=1,
                 n_conditions: int=1,
                 n_subjects: int=1) -> None:
        self.x_dim = x_dim
        super().__init__(stage_to_change="dimension reduction",
                         distribution_info={"x": None})
            
        extra_n_dim = np.sum([n for n in [n_batches, n_conditions, n_subjects] if n>1], dtype=int)
        
        self.loc_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=y_dim + extra_n_dim,
                              out_features=512,
                              bias=True)),
            ('fc1_bn', nn.BatchNorm1d(512)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=512,
                              out_features=256,
                              bias=True)),
            ('fc2_bn', nn.BatchNorm1d(256)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=256,
                              out_features=128,
                              bias=True)),
            ('fc3_bn', nn.BatchNorm1d(128)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=128,
                              out_features=x_dim,
                              bias=True))
        ]))
        self.log_scale_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=y_dim + extra_n_dim,
                              out_features=512,
                              bias=True)),
            ('fc1_bn', nn.BatchNorm1d(512)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=512,
                              out_features=256,
                              bias=True)),
            ('fc2_bn', nn.BatchNorm1d(256)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=256,
                              out_features=128,
                              bias=True)),
            ('fc3_bn', nn.BatchNorm1d(128)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=128,
                              out_features=x_dim,
                              bias=True))
        ]))
           
    def _update_distributions(self, 
                              z: torch.tensor, 
                              w: torch.tensor,
                              FB: torch.tensor,
                              FC: torch.tensor,
                              RS: torch.tensor):
        z_w = (z+w)/2
        
        effect_list = [m for m in [FB, FC, RS] if m.shape[1] > 1]
        
        z_w_with_effct = torch.cat([z_w] + effect_list, dim=1)
        
        self.distribution_dict['x'] = Independent(Normal(loc=self.loc_mapping(z_w_with_effct),
                                                         scale=F.softplus(self.log_scale_mapping(z_w_with_effct)) + 0.00001),
                                                         reinterpreted_batch_ndims=1)
        

    