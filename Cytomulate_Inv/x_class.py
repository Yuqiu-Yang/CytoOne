import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from collections import OrderedDict


class p_x_class(nn.Module):
    def __init__(self,
                 sigma_dict: dict={"x": 0.1}) -> None:
        super().__init__()
        
        self.scale = torch.tensor(sigma_dict['x'], dtype=torch.float32)
        
        self.distribution_dict = {"x": None}
    
    def _update_distribution(self, embedding):
        self.distribution_dict['x'] = Independent(Normal(loc=embedding,
                                                         scale=self.scale),
                                                         reinterpreted_batch_ndims=1)
    
    def get_sample(self):
        result_dict = {"x": None}
        result_dict['x'] = self.distribution_dict['x'].rsample()
        return result_dict
    
    def forward(self, embedding):
        self._update_distribution(embedding=embedding)
        return self.distribution_dict


class q_x_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int,
                 sigma_dict: dict={"x": 0.1}) -> None:
        super().__init__()
        
        self.loc_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=y_dim+3,
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
        
        self.scale = torch.tensor(sigma_dict['x'], dtype=torch.float32)
        
        self.distribution_dict = {"x": None}
        
        
    def _update_distribution(self, 
                             z: torch.tensor, 
                             w: torch.tensor,
                             FB: torch.tensor,
                             FC: torch.tensor,
                             RS: torch.tensor):
        b = torch.argmax(FB, dim=1)
        c = torch.argmax(FC, dim=1)
        s = torch.argmax(RS, dim=1)
        
        w_temp = torch.round(F.sigmoid(w))
        wq = w + (w_temp - w).detach()
        
        z = torch.exp(z)
        denoised_y = wq * z
        
        loc_map_input = torch.cat((denoised_y, b, c, s), dim=1)
        self.distribution_dict['x'] = Independent(Normal(loc=self.loc_mapping(loc_map_input),
                                                         scale=self.scale),
                                                         reinterpreted_batch_ndims=1)
        
    def get_sample(self):
        result_dict = {"x": None}
        result_dict['x'] = self.distribution_dict['x'].rsample()
        return result_dict
        
    def forward(self, 
                z: torch.tensor, 
                w: torch.tensor,
                FB: torch.tensor,
                FC: torch.tensor,
                RS: torch.tensor):
        self._update_distribution(z=z,
                                  w=w,
                                  FB=FB,
                                  FC=FC,
                                  RS=RS)
        return self.distribution_dict
        
    