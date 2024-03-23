import torch 
import torch.nn as nn 
import torch.nn.functional as F

from collections import OrderedDict


class x_posterior_module(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int=2,
                 scale: float=0.1) -> None:
        super().__init__()
        
        self.loc_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=y_dim,
                              out_features=512,
                              bias=True)),
            ('fc1_relu', nn.ReLU()),
            ('fc2', nn.Linear(in_features=512,
                              out_features=256,
                              bias=True)),
            ('fc2_relu', nn.ReLU()),
            ('fc3', nn.Linear(in_features=256,
                              out_features=128,
                              bias=True)),
            ('fc3_relu', nn.ReLU()),
            ('fc4', nn.Linear(in_features=128,
                              out_features=x_dim,
                              bias=True))
        ]))
        
        self.scale = torch.tensor(scale, dtype=torch.float32)
    
    def forward(self, w, z):
        w_temp = torch.round(F.sigmoid(w))
        wq = w + (w_temp - w).detach()
        
        z = torch.exp(z)
        denoised_y = wq * z
        
        return {
            'loc': self.loc_mapping(denoised_y),
            'scale': self.scale
        }
