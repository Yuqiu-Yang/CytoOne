import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from CytoOne.base_class import component_base_class

import numpy as np 

from collections import OrderedDict

from typing import Optional


class p_pretrain_z_class(component_base_class):
    def __init__(self) -> None:
        super().__init__(stage_to_change='pretrain', 
                         distribution_info={'z': None})
        self.loc = torch.tensor(0, dtype=torch.float32)
        self.scale = torch.tensor(5, dtype=torch.float32)

    def _update_distributions(self):
        self.distribution_dict['z'] = Independent(Normal(loc=self.loc,
                                                         scale=self.scale), 
                                                  reinterpreted_batch_ndims=0)  

   
class p_pretrain_w_class(component_base_class):
    def __init__(self) -> None:
        super().__init__(stage_to_change='pretrain', 
                         distribution_info={'w': None})
        self.loc = torch.tensor(0, dtype=torch.float32)
        self.scale = torch.tensor(5, dtype=torch.float32)

    def _update_distributions(self):
        self.distribution_dict['w'] = Independent(Normal(loc=self.loc,
                                                         scale=self.scale), 
                                                  reinterpreted_batch_ndims=0)   


class p_effect_z_w_class(component_base_class):
    def __init__(self):
        super().__init__(stage_to_change="expression effect estimation",
                         distribution_info={'z_w': None})
        
    def _update_distributions(self,
                              one_hot_index: torch.tensor,
                              FB: torch.tensor,
                              FC: torch.tensor,
                              RS: torch.tensor, 
                              theta_loc_sample: torch.tensor,
                              alpha_loc_sample: torch.tensor,
                              beta_loc_sample: torch.tensor,
                              gamma_loc_sample: torch.tensor,
                              alpha_theta_loc_sample: torch.tensor,
                              beta_theta_loc_sample: torch.tensor,
                              theta_scale_sample: torch.tensor,
                              alpha_scale_sample: torch.tensor,
                              beta_scale_sample: torch.tensor,
                              gamma_scale_sample: torch.tensor,
                              alpha_theta_scale_sample: torch.tensor,
                              beta_theta_scale_sample: torch.tensor):
        FB_theta = torch.cat([one_hot_index[:, [c]] * FB for c in range(one_hot_index.shape[1])],
                             dim=1)
        FC_theta = torch.cat([one_hot_index[:, [c]] * FC for c in range(one_hot_index.shape[1])],
                             dim=1)
        
        loc = torch.matmul(one_hot_index, theta_loc_sample) + \
              torch.matmul(FB, alpha_loc_sample) + \
              torch.matmul(FC, beta_loc_sample) + \
              torch.matmul(RS, gamma_loc_sample) + \
              torch.matmul(FB_theta, alpha_theta_loc_sample) + \
              torch.matmul(FC_theta, beta_theta_loc_sample) 
              
        log_scale = torch.matmul(one_hot_index, theta_scale_sample) + \
                    torch.matmul(FB, alpha_scale_sample) + \
                    torch.matmul(FC, beta_scale_sample) + \
                    torch.matmul(RS, gamma_scale_sample) + \
                    torch.matmul(FB_theta, alpha_theta_scale_sample) + \
                    torch.matmul(FC_theta, beta_theta_scale_sample) 

        self.distribution_dict['z_w'] = Independent(Normal(loc=loc,
                                                         scale=F.softplus(log_scale, beta=1) + 0.00001),
                                                  reinterpreted_batch_ndims=1)
        


class p_z_w_class(component_base_class):
    def __init__(self,
                 y_dim: int,
                 x_dim: int,
                 n_batches: int=1,
                 n_conditions: int=1,
                 n_subjects: int=1) -> None:
        super().__init__(stage_to_change="dimension reduction",
                         distribution_info={"z": None,
                                            "w": None})
        
        extra_n_dim = np.sum([n for n in [n_batches, n_conditions, n_subjects] if n>1], dtype=int)
        self.mu_z_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=x_dim + extra_n_dim,
                                out_features=128,
                                bias=True)),
            ('fc1_bn', nn.BatchNorm1d(128)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=128,
                                out_features=256,
                                bias=True)),
            ('fc2_bn', nn.BatchNorm1d(256)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=256,
                                out_features=512,
                                bias=True)),
            ('fc3_bn', nn.BatchNorm1d(512)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=512,
                                out_features=y_dim,
                                bias=True))
        ]))
        
        self.log_Sigma_z_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=x_dim + extra_n_dim,
                                out_features=128,
                                bias=True)),
            ('fc1_bn', nn.BatchNorm1d(128)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=128,
                                out_features=256,
                                bias=True)),
            ('fc2_bn', nn.BatchNorm1d(256)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=256,
                                out_features=512,
                                bias=True)),
            ('fc3_bn', nn.BatchNorm1d(512)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=512,
                                out_features=y_dim,
                                bias=True))
        ]))
        
        self.mu_w_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=x_dim + extra_n_dim,
                            out_features=128,
                            bias=True)),
            ('fc1_bn', nn.BatchNorm1d(128)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=128,
                                out_features=256,
                                bias=True)),
            ('fc2_bn', nn.BatchNorm1d(256)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=256,
                                out_features=512,
                                bias=True)),
            ('fc3_bn', nn.BatchNorm1d(512)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=512,
                                out_features=y_dim,
                                bias=True))
        ]))
        
        self.log_Sigma_w_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=x_dim + extra_n_dim,
                                out_features=128,
                                bias=True)),
            ('fc1_bn', nn.BatchNorm1d(128)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=128,
                                out_features=256,
                                bias=True)),
            ('fc2_bn', nn.BatchNorm1d(256)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=256,
                                out_features=512,
                                bias=True)),
            ('fc3_bn', nn.BatchNorm1d(512)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=512,
                                out_features=y_dim,
                                bias=True))
        ]))
                    
    def _update_distributions(self,
                              x: torch.tensor,
                              FB: torch.tensor,
                              FC: torch.tensor,
                              RS: torch.tensor):
        
        effect_list = [m for m in [FB, FC, RS] if m.shape[1] > 1]
        
        x_with_effect = torch.cat([x]+ effect_list, dim=1)
        
        z_loc = self.mu_z_mapping(x_with_effect) 
        # z_scale = torch.tensor(0.1, dtype=torch.float32)
        z_scale = F.softplus(self.log_Sigma_z_mapping(x_with_effect), beta=1) + 0.00001
        w_loc = self.mu_w_mapping(x_with_effect) 
        # w_scale = torch.tensor(0.1, dtype=torch.float32)
        w_scale = F.softplus(self.log_Sigma_w_mapping(x_with_effect), beta=1) + 0.00001
        self.distribution_dict['z'] = Independent(Normal(loc=z_loc,
                                                         scale=z_scale),
                                                    reinterpreted_batch_ndims=1)
        self.distribution_dict['w'] = Independent(Normal(loc=w_loc,
                                                         scale=w_scale),
                                                    reinterpreted_batch_ndims=1)


class q_pretrain_z_class(component_base_class):
    def __init__(self,
                 y_dim: int=1) -> None:
        super().__init__(stage_to_change='pretrain', 
                         distribution_info={'z': None})
        self.y_dim = y_dim
        self.reinterpreted_batch_ndims = 0
        if self.y_dim > 1:
            self.reinterpreted_batch_ndims = 1
            
        self.mu_z_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=1,
                                out_features=16,
                                bias=True)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=16,
                                out_features=8,
                                bias=True)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=8,
                                out_features=4,
                                bias=True)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=4,
                                out_features=1,
                                bias=True))
        ]))
        
        self.log_Sigma_z_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=1,
                                out_features=16,
                                bias=True)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=16,
                                out_features=8,
                                bias=True)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=8,
                                out_features=4,
                                bias=True)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=4,
                                out_features=1,
                                bias=True))
        ]))
    
    def _update_distributions(self,
                              y: torch.tensor):        
        flattened_y = y.view((-1, 1))
        
        loc = self.mu_z_mapping(flattened_y)
        scale = F.softplus(self.log_Sigma_z_mapping(flattened_y)) + 0.00001
        
        # reshape z_loc and z_scale 
        unflattened_loc = loc.view((-1, self.y_dim))
        unflattened_scale = scale.view((-1, self.y_dim))
        
        self.distribution_dict['z'] = Independent(Normal(loc=unflattened_loc,
                                                         scale=unflattened_scale),
                                                  reinterpreted_batch_ndims=self.reinterpreted_batch_ndims)
        
    
class q_pretrain_w_class(component_base_class):
    def __init__(self,
                 y_dim: int=1) -> None:
        super().__init__(stage_to_change='pretrain', 
                         distribution_info={'w': None})
        self.y_dim = y_dim
        self.reinterpreted_batch_ndims = 0
        if self.y_dim > 1:
            self.reinterpreted_batch_ndims = 1
        
        self.mu_w_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=1,
                                out_features=16,
                                bias=True)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=16,
                                out_features=8,
                                bias=True)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=8,
                                out_features=4,
                                bias=True)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=4,
                                out_features=1,
                                bias=True))
        ]))
        
        self.log_Sigma_w_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=1,
                                out_features=16,
                                bias=True)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=16,
                                out_features=8,
                                bias=True)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=8,
                                out_features=4,
                                bias=True)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=4,
                                out_features=1,
                                bias=True))
        ]))
    
    def _update_distributions(self,
                              y: torch.tensor):
        flattened_y = y.view((-1, 1))
        
        loc = self.mu_w_mapping(flattened_y)
        scale = F.softplus(self.log_Sigma_w_mapping(flattened_y)) + 0.00001
        
        # reshape z_loc and z_scale 
        unflattened_loc = loc.view((-1, self.y_dim))
        unflattened_scale = scale.view((-1, self.y_dim))
        
        self.distribution_dict['w'] = Independent(Normal(loc=unflattened_loc,
                                                         scale=unflattened_scale),
                                                  reinterpreted_batch_ndims=self.reinterpreted_batch_ndims)
    

class q_z_w_class(component_base_class):
    def __init__(self,
                 y_dim: int) -> None:
        super().__init__(stage_to_change="pretrain",
                         distribution_info={"z": None,
                                            "w": None})
        self.y_dim = y_dim
        self.q_z = q_pretrain_z_class(y_dim=y_dim)
        self.q_w = q_pretrain_w_class(y_dim=y_dim)
    
    def load_pretrained_model(self,
                              model_check_point_path: str) -> None:
        ckpt = torch.load(model_check_point_path)
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        # pretrained_dict = {k.replace("q_z.", "").replace("q_w.", ""): v for k, v in ckpt['model_state_dict'].items() if (k.replace("q_z.", "") in model_dict) or (k.replace("q_w.", "") in model_dict)}
        pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() if (k in model_dict)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        self.load_state_dict(pretrained_dict)
        # 3. Then, we freeze the parameters 
        for param in self.parameters():
            param.requires_grad = False
        
    def _update_distributions(self,
                              y: torch.tensor):
        self.distribution_dict['z'] = self.q_z(y=y)['z']
        self.distribution_dict['w'] = self.q_w(y=y)['w']


