import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from collections import OrderedDict
                                
from pyro.distributions import Delta 


class p_w_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int,
                 distribution_type: str="ZILN") -> None:
        super().__init__()
        
        assert distribution_type in ["ZILN", "N", "MU"], "distribution_type has to be one of ZILN, N, or MU"
        self.distribution_type = distribution_type
        
        if distribution_type == "ZILN":
            # When the distribution is Zero Inflated 
            # w is Delta at 1
            # The Delta distribution in pyro only 
            # has two parameters 
            self.v_mapping = nn.Linear(in_features=x_dim,
                                       out_features=y_dim)
            self.log_density_mapping = nn.Linear(in_features=x_dim,
                                       out_features=y_dim)
            # Zeros out all the weights 
            self.v_mapping.weight.data.zero_()
            self.log_density_mapping.weight.data.zero_()
            # Set bias to 1
            self.v_mapping.bias.data.fill_(1.0)
            self.log_density_mapping.bias.data.fill_(1.0)
            # Set requires_grad to False 
            for param in self.v_mapping.parameters():
                param.requires_grad = False 
            for param in self.log_density_mapping.parameters():
                param.requires_grad = False  
            
        elif distribution_type in ["N", "MU"]:
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
        else:
            raise NotImplementedError
        
        self.distribution_dict = {"w": None}
        
    def _update_distribution(self,
                x: torch.tensor,
                alpha_w_mu_sample: torch.tensor,
                beta_w_mu_sample: torch.tensor,
                gamma_w_mu_sample: torch.tensor,
                FB: torch.tensor,
                FC: torch.tensor,
                RS: torch.tensor):
        if self.distribution_type == "ZILN":
            self.distribution_dict['w'] = Independent(Delta(v=self.v_mapping(x),
                                                            log_density=self.log_density_mapping(x)),
                                                        reinterpreted_batch_ndims=1) 
        else: 
            loc = self.loc_mapping(x) + \
                  torch.einsum("nb, bmd, nd -> nm", FB, alpha_w_mu_sample, x) + \
                  torch.einsum("nc, cmd, nd -> nm", FC, beta_w_mu_sample, x) + \
                  torch.einsum("ns, smd, nd -> nm", RS, gamma_w_mu_sample, x)
            scale = F.softplus(self.scale_mapping(x), beta=1) + 0.00001
            self.distribution_dict['w'] = Independent(Normal(loc=loc,
                                                             scale=scale),
                                                      reinterpreted_batch_ndims=1)

    def get_sample(self):
        result_dict = {"w": None}
        result_dict['w'] = self.distribution_dict['w'].rsample()
        return result_dict 
        
    def forward(self,
                x: torch.tensor,
                alpha_w_mu_sample: torch.tensor,
                beta_w_mu_sample: torch.tensor,
                gamma_w_mu_sample: torch.tensor,
                FB: torch.tensor,
                FC: torch.tensor,
                RS: torch.tensor):
        self._update_distribution(x=x, 
                                  alpha_w_mu_sample=alpha_w_mu_sample,
                                  beta_w_mu_sample=beta_w_mu_sample,
                                  gamma_w_mu_sample=gamma_w_mu_sample,
                                  FB=FB,
                                  FC=FC,
                                  RS=RS)
        return self.distribution_dict 


class q_w_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 distribution_type: str="ZILN") -> None:
        super().__init__()
        
        self.distribution_type = distribution_type
        
        if distribution_type == "ZILN":
            # When the distribution is Zero Inflated 
            # w is Delta at 1
            # The Delta distribution in pyro only 
            # has two parameters 
            self.v_mapping = nn.Linear(in_features=y_dim,
                                       out_features=y_dim)
            self.log_density_mapping = nn.Linear(in_features=y_dim,
                                       out_features=y_dim)
            # Zeros out all the weights 
            self.v_mapping.weight.data.zero_()
            self.log_density_mapping.weight.data.zero_()
            # Set bias to 1
            self.v_mapping.bias.data.fill_(1.0)
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
    
        self.distribution_dict = {"w": None}    
    
    def _update_distribution(self,
                             y: torch.tensor):
        if self.distribution_type == "ZILN":
            self.distribution_dict['w'] = Independent(Delta(v=self.v_mapping(y),
                                                            log_density=self.log_density_mapping(y)),
                                                        reinterpreted_batch_ndims=1) 
        if self.distribution_type in ["N", "MU"]:
            loc = self.loc_mapping(y) 
            scale = F.softplus(self.scale_mapping(y), beta=10) + 0.00001
            self.distribution_dict['w'] = Independent(Normal(loc=loc,
                                                             scale=scale),
                                                      reinterpreted_batch_ndims=1)
    
    def get_sample(self):
        result_dict = {"w": None}
        result_dict['w'] = self.distribution_dict['w'].rsample()
        return result_dict  
    
    def forward(self,
                y: torch.tensor):
        self._update_distribution(y=y)
        return self.distribution_dict 




