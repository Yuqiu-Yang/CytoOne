import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from collections import OrderedDict

from CytoOne.basic_distributions import zero_inflated_lognormal
                                
from pyro.distributions import Delta 


class p_z_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int=2,
                 distribution_type: str="ZILN") -> None:
        super().__init__()
        self.distribution_type = distribution_type
        self.loc_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=x_dim,
                                out_features=128,
                                bias=True)),
            ('fc1_relu', nn.ReLU()),
            ('fc2', nn.Linear(in_features=128,
                                out_features=256,
                                bias=True)),
            ('fc2_relu', nn.ReLU()),
            ('fc3', nn.Linear(in_features=256,
                                out_features=512,
                                bias=True)),
            ('fc3_relu', nn.ReLU()),
            ('fc4', nn.Linear(in_features=512,
                                out_features=y_dim,
                                bias=True))
        ]))
        
        self.scale_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=x_dim,
                                out_features=128,
                                bias=True)),
            ('fc1_relu', nn.ReLU()),
            ('fc2', nn.Linear(in_features=128,
                                out_features=256,
                                bias=True)),
            ('fc2_relu', nn.ReLU()),
            ('fc3', nn.Linear(in_features=256,
                                out_features=512,
                                bias=True)),
            ('fc3_relu', nn.ReLU()),
            ('fc4', nn.Linear(in_features=512,
                                out_features=y_dim,
                                bias=True))
        ]))
        
        if distribution_type == "ZILN":
            # When the distribution is Zero Inflated 
            # z is ZILN
            self.gate_mapping = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(in_features=x_dim,
                                    out_features=128,
                                    bias=True)),
                ('fc1_relu', nn.ReLU()),
                ('fc2', nn.Linear(in_features=128,
                                    out_features=256,
                                    bias=True)),
                ('fc2_relu', nn.ReLU()),
                ('fc3', nn.Linear(in_features=256,
                                    out_features=512,
                                    bias=True)),
                ('fc3_relu', nn.ReLU()),
                ('fc4', nn.Linear(in_features=512,
                                    out_features=y_dim,
                                    bias=True))
        ]))
            
        elif distribution_type not in ["N", "MU"]:
            raise NotImplementedError
        
        self.distribution_dict = {"z": None}
        
    def _update_distribution(self):
        pass 
    
    def get_sample(self):
        result_dict = {"z": None}
        result_dict['z'] = self.distribution_dict['z'].rsample()
        return result_dict  
    
    def forward(self):
        pass 


class q_z_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 distribution_type: str="ZILN") -> None:
        super().__init__()
        
        self.distribution_type = distribution_type
        
        if distribution_type == "ZILN":
            # When the distribution is Zero Inflated 
            # z is ZILN
            self.v_mapping = nn.Linear(in_features=y_dim,
                                       out_features=y_dim,
                                       bias=False)
            self.log_density_mapping = nn.Linear(in_features=y_dim,
                                       out_features=y_dim)
            # Zeros out all the weights 
            self.v_mapping.weight.data.copy_(torch.eye(y_dim))
            self.log_density_mapping.weight.data.zero_()
            # Set bias to 1
            self.log_density_mapping.bias.data.fill_(1.0)
            # Set requires_grad to False 
            for param in self.v_mapping.parameters():
                param.requires_grad = False 
            for param in self.log_density_mapping.parameters():
                param.requires_grad = False 
            
        elif distribution_type in ["N", "MU"]:
            self.loc_mapping = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(in_features=y_dim,
                                    out_features=y_dim))
            ]))
            
            self.scale_mapping = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(in_features=y_dim,
                                    out_features=y_dim))
            ]))
        else:
            raise NotImplementedError
    
        self.distribution_dict = {"z": None}

        
    def _update_distribution(self,
                             y: torch.tensor):
        if self.distribution_type == "ZILN":
            return {
                'v': self.v_mapping(y),
                'log_density': self.log_density_mapping(y)
            }

        if self.distribution_type in ["N", "MU"]:
            return {
                'loc': self.loc_mapping(y),
                'scale': F.softplus(self.scale_mapping(y), beta=10) + 0.00001
            } 
    
    def get_sample(self):
        result_dict = {"z": None}
        result_dict['z'] = self.distribution_dict['z'].rsample()
        return result_dict 
    
    def forward(self,
                y: torch.tensor):
        self._update_distribution(y=y)
        return self.distribution_dict 




class manifestation_prior_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int=2,
                 distribution_type: str="ZILN") -> None:
        super().__init__()
        
        self.distribution_type = distribution_type
        self.w_module = w_prior_module(y_dim=y_dim,
                                       x_dim=x_dim,
                                       distribution_type=distribution_type)
        self.z_module = z_prior_module(y_dim=y_dim,
                                       x_dim=x_dim,
                                       distribution_type=distribution_type)
        
    def forward(self, x):
        # The forward passes in priors only 
        # need to output distributions 
        w_dict = self.w_module(x=x)
        z_dict = self.z_module(x=x)
        if self.distribution_type == "ZILN":
            return {
                'w_prior': Independent(Delta(**w_dict),
                                       reinterpreted_batch_ndims=1),
                'z_prior': Independent(zero_inflated_lognormal(**z_dict),
                                       reinterpreted_batch_ndims=1)
            }
        if self.distribution_type in ["N", "MU"]:
            return {
                'w_prior': Independent(Normal(**w_dict),
                                       reinterpreted_batch_ndims=1),
                'z_prior': Independent(Normal(**z_dict),
                                       reinterpreted_batch_ndims=1)
            }


class manifestation_posterior_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 distribution_type: str="ZILN") -> None:
        super().__init__()
        self.distribution_type = distribution_type
        self.w_module = w_posterior_module(y_dim=y_dim,
                                            distribution_type=distribution_type)
        self.z_module = z_posterior_module(y_dim=y_dim,
                                            distribution_type=distribution_type)
        
        
    def forward(self, y):
        w_dict = self.w_module(y=y)
        z_dict = self.z_module(y=y)
        if self.distribution_type == "ZILN":
            return {
                'w_posterior': Independent(Delta(**w_dict),
                                       reinterpreted_batch_ndims=1),
                'z_posterior': Independent(Delta(**z_dict),
                                       reinterpreted_batch_ndims=1)
            }
        if self.distribution_type in ["N", "MU"]:
            return {
                'w_posterior': Independent(Normal(**w_dict),
                                       reinterpreted_batch_ndims=1),
                'z_posterior': Independent(Normal(**z_dict),
                                       reinterpreted_batch_ndims=1)
            }
