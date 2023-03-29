import torch 
import torch.nn as nn 
import torch.nn.functional as F

from collections import OrderedDict


class x_posterior_module(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int=2) -> None:
        super().__init__()
        
        self.loc_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=y_dim,
                              out_features=x_dim))
        ]))
        
        self.scale_mapping = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(in_features=y_dim,
                              out_features=x_dim)),
            ('softplus', nn.Softplus())
        ]))
    
    def forward(self, w, z):
        w_temp = torch.round(F.sigmoid(w))
        wq = w + (w_temp - w).detach()
        
        z = torch.exp(z)
        denoised_y = wq * z
        
        return {
            'loc': self.loc_mapping(denoised_y),
            'scale': self.scale_mapping(denoised_y)
        }
