import torch 
import torch.nn as nn 
from torch.distributions import Normal, Independent

from Cytomulate_Inv.network_modules.bottleneck_posterior_modules import x_posterior_module

class bottleneck_prior_class(nn.Module):
    def __init__(self,
                 scale: float=0.1) -> None:
        super().__init__()
        self.scale = scale
        
    def forward(self, embedding):
        return {
            "x_prior": Independent(Normal(loc=embedding, scale=self.scale),
                                    reinterpreted_batch_ndims=1)
        }


class bottleneck_posterior_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int=2) -> None:
        super().__init__()
        self.x_module = x_posterior_module(y_dim=y_dim,
                                           x_dim=x_dim)
        
    def forward(self, w, z):
        x_dict = self.x_module(w=w, z=z) 
        return {
            'x_posterior': Independent(Normal(**x_dict),
                                       reinterpreted_batch_ndims=1)
        }
    