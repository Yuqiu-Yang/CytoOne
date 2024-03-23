import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from CytoOne.base_class import component_series_class

import numpy as np 
from collections import OrderedDict

from typing import Union, Optional


class p_x_class(component_series_class):
    def __init__(self, 
                 x_dim: Union[int, list]) -> None:
        # We assume x_dim is ordered T, T-1, ..., 0
        if isinstance(x_dim, int):
            x_dim = [x_dim]
        self.x_dim = x_dim
        super().__init__(var_name="x", 
                         series_length=len(self.x_dim), 
                         sample_from='T',
                         stage_to_change="dimension reduction")
        
        self.loc_mapping = nn.ModuleDict({})
        self.log_scale = nn.ParameterDict({})
        if self.T > 0:
            for n in range(self.T-1, -1, -1):
                self.loc_mapping['x'+str(n)] = nn.Linear(in_features=self.x_dim[self.T-n-1],
                                                         out_features=self.x_dim[self.T-n],
                                                         bias=False)
                self.log_scale['x'+str(n)] = nn.Parameter(torch.randn(1), requires_grad=True)
        
    def _update_distributions(self,
                              embedding: Optional[torch.tensor]=None,
                              q_x_samples: Optional[dict]=None,
                              get_mean: bool=False):
        if embedding is None:
            self.distribution_dict["x"+str(self.T)] = Independent(Normal(loc=torch.zeros(self.x_dim[0]),
                                                                         scale=torch.ones(self.x_dim[0])), 
                                                                  reinterpreted_batch_ndims=1) 
        else: 
            self.distribution_dict["x"+str(self.T)] = Independent(Normal(loc=embedding,
                                                                         scale=torch.ones(self.x_dim[0])), 
                                                                  reinterpreted_batch_ndims=1) 
        if get_mean:
            self.samples["x"+str(self.T)] = self.distribution_dict['x'+str(self.T)].mean
        else:
            self.samples["x"+str(self.T)] = self.distribution_dict['x'+str(self.T)].rsample()
        if self.T > 0:
            for n in range(self.T-1, -1, -1):
                if q_x_samples is None:
                    self.distribution_dict["x"+str(n)] = Independent(Normal(loc=self.loc_mapping['x'+str(n)](self.samples["x"+str(n+1)]),
                                                                            scale=F.softplus(self.log_scale['x'+str(n)]) + 0.00001), 
                                                                    reinterpreted_batch_ndims=1)
                else:
                    self.distribution_dict["x"+str(n)] = Independent(Normal(loc=self.loc_mapping['x'+str(n)](q_x_samples["x"+str(n+1)]),
                                                                            scale=F.softplus(self.log_scale['x'+str(n)]) + 0.00001), 
                                                                    reinterpreted_batch_ndims=1)
                if get_mean:
                    self.samples["x"+str(n)] = self.distribution_dict['x'+str(n)].mean
                else:    
                    self.samples["x"+str(n)] = self.distribution_dict['x'+str(n)].rsample()
    
        
class q_x_class(component_series_class):
    def __init__(self,
                 y_dim: int,
                 x_dim: Union[int, list],
                 n_batches: int=1,
                 n_conditions: int=1,
                 n_subjects: int=1) -> None:
        # We assume x_dim is ordered T, T-1, ..., 0
        if isinstance(x_dim, int):
            x_dim = [x_dim]
        self.x_dim = x_dim
        super().__init__(var_name="x", 
                         series_length=len(self.x_dim),
                         sample_from="0",
                         stage_to_change="dimension reduction")
            
        extra_n_dim = np.sum([n for n in [n_batches, n_conditions, n_subjects] if n>1], dtype=int)
        
        self.loc_mapping = nn.ModuleDict({"x0": nn.Sequential(OrderedDict([
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
                              out_features=x_dim[-1],
                              bias=True))
        ]))})
        self.log_scale_mapping = nn.ModuleDict({"x0": nn.Sequential(OrderedDict([
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
                              out_features=x_dim[-1],
                              bias=True))
        ]))})
        
        if self.T > 0:
            for n in range(1, self.T+1):
                self.loc_mapping['x'+str(n)] = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(in_features=self.x_dim[self.T-n+1],
                                    out_features=self.x_dim[self.T-n],
                                    bias=False))
                ]))
                self.log_scale_mapping['x'+str(n)] = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(in_features=self.x_dim[self.T-n+1],
                                    out_features=self.x_dim[self.T-n],
                                    bias=False))
                ]))
                
    def _update_distributions(self, 
                              z: torch.tensor, 
                              w: torch.tensor,
                              FB: torch.tensor,
                              FC: torch.tensor,
                              RS: torch.tensor,
                              get_mean: bool=False):
        z_w = (z+w)/2
        
        effect_list = [m for m in [FB, FC, RS] if m.shape[1] > 1]
        
        z_w_with_effct = torch.cat([z_w] + effect_list, dim=1)
        
        self.distribution_dict['x0'] = Independent(Normal(loc=self.loc_mapping['x0'](z_w_with_effct),
                                                         scale=F.softplus(self.log_scale_mapping['x0'](z_w_with_effct)) + 0.00001),
                                                         reinterpreted_batch_ndims=1)
        if get_mean:
            self.samples["x0"] = self.distribution_dict['x0'].mean
        else: 
            self.samples["x0"] = self.distribution_dict['x0'].rsample()
        if self.T > 0:
            for n in range(1, self.T+1):
                self.distribution_dict["x"+str(n)] = Independent(Normal(loc=self.loc_mapping['x'+str(n)](self.samples["x"+str(n-1)]),
                                                                        scale=F.softplus(self.log_scale_mapping['x'+str(n)](self.samples["x"+str(n-1)])) + 0.00001), 
                                                                reinterpreted_batch_ndims=1)
                if get_mean:
                    self.samples["x"+str(n)] = self.distribution_dict['x'+str(n)].mean
                else:
                    self.samples["x"+str(n)] = self.distribution_dict['x'+str(n)].rsample()
    
        

# class p_x_class(component_base_class):
#     def __init__(self, 
#                  x_dim: int) -> None:
#         super().__init__(stage_to_change="dimension reduction",
#                          distribution_info={"x": None})
#         self.x_dim = x_dim
    
#     def _update_distributions(self):
#         self.distribution_dict['x'] = Independent(Normal(loc=torch.zeros(self.x_dim),
#                                                          scale=torch.ones(self.x_dim)), 
#                                                   reinterpreted_batch_ndims=1)   


# class p_x_vq_vae_class(component_base_class):
#     def __init__(self,
#                  x_dim: int) -> None:
#         super().__init__(stage_to_change="clustering",
#                          distribution_info={"x": None})
        
#         self.x_dim = x_dim 
                
#     def _update_distributions(self, 
#                               embedding: torch.tensor):
#         self.distribution_dict['x'] = Independent(Normal(loc=embedding,
#                                                          scale=torch.ones(self.x_dim)),
#                                                          reinterpreted_batch_ndims=1)


# class q_x_class(component_base_class):
#     def __init__(self,
#                  y_dim: int,
#                  x_dim: int,
#                  n_batches: int=1,
#                  n_conditions: int=1,
#                  n_subjects: int=1) -> None:
#         super().__init__(stage_to_change="dimension reduction",
#                          distribution_info={"x": None})
        
#         extra_n_dim = np.sum([n for n in [n_batches, n_conditions, n_subjects] if n>1], dtype=int)
#         self.loc_mapping = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(in_features=y_dim + extra_n_dim,
#                               out_features=512,
#                               bias=True)),
#             ('fc1_bn', nn.BatchNorm1d(512)),
#             ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
#             ('fc2', nn.Linear(in_features=512,
#                               out_features=256,
#                               bias=True)),
#             ('fc2_bn', nn.BatchNorm1d(256)),
#             ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
#             ('fc3', nn.Linear(in_features=256,
#                               out_features=128,
#                               bias=True)),
#             ('fc3_bn', nn.BatchNorm1d(128)),
#             ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
#             ('fc4', nn.Linear(in_features=128,
#                               out_features=x_dim,
#                               bias=True))
#         ]))
        
#         self.log_scale_mapping = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(in_features=y_dim + extra_n_dim,
#                               out_features=512,
#                               bias=True)),
#             ('fc1_bn', nn.BatchNorm1d(512)),
#             ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
#             ('fc2', nn.Linear(in_features=512,
#                               out_features=256,
#                               bias=True)),
#             ('fc2_bn', nn.BatchNorm1d(256)),
#             ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
#             ('fc3', nn.Linear(in_features=256,
#                               out_features=128,
#                               bias=True)),
#             ('fc3_bn', nn.BatchNorm1d(128)),
#             ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
#             ('fc4', nn.Linear(in_features=128,
#                               out_features=x_dim,
#                               bias=True))
#         ]))
        
                    
#     def _update_distributions(self, 
#                               z: torch.tensor, 
#                               w: torch.tensor,
#                               FB: torch.tensor,
#                               FC: torch.tensor,
#                               RS: torch.tensor):
#         z_w = (z+w)/2
        
#         effect_list = [m for m in [FB, FC, RS] if m.shape[1] > 1]
        
#         z_w_with_effct = torch.cat([z_w] + effect_list, dim=1)
#         loc = self.loc_mapping(z_w_with_effct)
#         log_scale = self.log_scale_mapping(z_w_with_effct)
        
#         self.distribution_dict['x'] = Independent(Normal(loc=loc,
#                                                          scale=F.softplus(log_scale) + 0.00001),
#                                                          reinterpreted_batch_ndims=1)
    
    
    