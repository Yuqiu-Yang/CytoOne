import torch 
import torch.nn as nn 
import torch.nn.functional as F

from collections import OrderedDict

class w_posterior_module(nn.Module):
    def __init__(self,
                 y_dim: int,
                 distribution_type: str="ZILN"
                 ) -> None:
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
                                  out_features=y_dim)),
                ('softplus', nn.Softplus())
            ]))
        else:
            raise NotImplementedError
        
    def forward(self, y):
        if self.distribution_type == "ZILN":
            return {
                'v': self.v_mapping(y),
                'log_density': self.log_density_mapping(y)
            }
        if self.distribution_type in ["N", "MU"]:
            return {
                'loc': self.loc_mapping(y),
                'scale': self.scale_mapping(y)
            }


class z_posterior_module(nn.Module):
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
                                    out_features=y_dim)),
                ('softplus', nn.Softplus())
            ]))
        else:
            raise NotImplementedError
        
    def forward(self, y):
        if self.distribution_type == "ZILN":
            return {
                'v': self.v_mapping(y),
                'log_density': self.log_density_mapping(y)
            }

        if self.distribution_type in ["N", "MU"]:
            return {
                'loc': self.loc_mapping(y),
                'scale': self.scale_mapping(y)
            }
        

