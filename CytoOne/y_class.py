import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from CytoOne.base_class import component_base_class


class p_y_class(component_base_class):
    def __init__(self,
                 y_dim: int=1,
                 y_scale: float=0.01) -> None:
        super().__init__(stage_to_change="pretrain",
                         distribution_info={"y": None})
        self.y_dim = y_dim
        self.y_scale = y_scale
        self.reinterpreted_batch_ndims = 0
        if self.y_dim > 1:
            self.reinterpreted_batch_ndims = 1
    
    def _update_distributions(self,
                              z: torch.tensor,
                              w: torch.tensor):
        
        w_temp = torch.round(F.sigmoid(w))
        wq = w + (w_temp - w).detach()
        
        z = torch.exp(z)
        denoised_y = wq * z 
        
        self.distribution_dict['y'] = Independent(Normal(loc=denoised_y,
                                                         scale=torch.tensor(self.y_scale, dtype=torch.float32)),
                                                  reinterpreted_batch_ndims=self.reinterpreted_batch_ndims)

        