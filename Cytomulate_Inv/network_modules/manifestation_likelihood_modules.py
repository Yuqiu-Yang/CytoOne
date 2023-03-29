import torch 
import torch.nn as nn 
import torch.nn.functional as F

from collections import OrderedDict

class y_module(nn.Module):
    def __init__(self,
                 y_dim: int,
                 distribution_type: str="ZILN",
                 mu_normal_scale: float=0.01) -> None:
        super().__init__()
        self.distribution_type = distribution_type
        
        self.identity_mapping = nn.Linear(in_features=y_dim,
                                       out_features=y_dim,
                                       bias=False)
        # Set v_mapping to identity mapping 
        self.identity_mapping.weight.data.copy_(torch.eye(y_dim))

        self.one_mapping = nn.Linear(in_features=y_dim,
                                       out_features=y_dim)
        # Zero out weight and set bias to 1
        self.one_mapping.weight.data.zero_()
        self.one_mapping.bias.data.fill_(1.0)
        for param in self.identity_mapping.parameters():
            param.requires_grad = False 
        for param in self.one_mapping.parameters():
            param.requires_grad = False 
            
        if distribution_type == "ZILN":
            # When the distribution is Zero Inflated 
            # y is Delta at sigmoid[w] * exp(z)
            # The Delta distribution in pyro only 
            # has two parameters 
            pass 
        elif self.distribution_type == "N":
            self.scale = F.softplus(nn.Parameter(torch.randn(1)))
        elif self.distribution_type == "MU":
            self.scale = F.softplus(nn.Parameter(mu_normal_scale, requires_grad=False))
            self.l = -F.softplus(nn.Parameter(torch.randn(1)))
        else:
            raise NotImplementedError
        
    def forward(self, w, z):
        w_temp = torch.round(F.sigmoid(w))
        wq = w + (w_temp - w).detach()
        
        z = torch.exp(z)
        loc = wq * z
        
        if self.distribution_type == "ZILN":
            return {
                'v': self.identity_mapping(loc),
                'log_density': self.one_mapping(loc)
            }
        if self.distribution_type == "N":
            return {
                'loc': self.identity_mapping(loc),
                'scale': self.scale
            }
        if self.distribution_type == "MU":
            return {
                'uniform_low': self.identity_mapping(loc) + self.l,
                'uniform_up': self.identity_mapping(loc),
                'normal_scale': self.scale
            }
        
        
    
    

