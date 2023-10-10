import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal
from pyro.distributions import Delta 


class p_beta_class(nn.Module):
    def __init__(self,
                 y_dim: int, 
                 x_dim: int,
                 n_conditions: int,
                 n_cell_types: int, 
                 sigma_dict: dict={"pi": 0.1,
                                   "z_mu": 0.1,
                                   "z_Sigma": 0.1,
                                   "w_mu": 0.1}) -> None:
        super().__init__()
        
        self.parameter_dict = {"pi": {"Delta": {},
                                      "Normal": {}},
                               "z_mu": {"Delta": {},
                                        "Normal": {}},
                               "z_Sigma": {"Delta": {},
                                           "Normal": {}},
                               "w_mu": {"Delta": {},
                                        "Normal": {}}}
        
        self.distribution_dict = {"pi": {},
                                  "z_mu": {},
                                  "z_Sigma": {},
                                  "w_mu": {}}

        for dist in self.parameter_dict:
            # The first distribution is Delta centered at 0 
            if dist == "pi":
                delta_size = (1, n_cell_types)
                normal_size = (n_conditions-1, n_cell_types)
            else:
                delta_size = (1, y_dim, x_dim)
                normal_size = (n_conditions-1, y_dim, x_dim)
                
            self.parameter_dict[dist]['Delta']['v'] = nn.Parameter(torch.zeros(delta_size),
                                                                    requires_grad=False)
            self.parameter_dict[dist]['Delta']['log_density'] = nn.Parameter(torch.ones(delta_size),
                                                                                requires_grad=False)
            self.distribution_dict[dist]['Delta'] = Delta(v=self.parameter_dict[dist]['Delta']['v'],
                                                          log_density=self.parameter_dict[dist]['Delta']['log_density'])
            self.parameter_dict[dist]['Normal']['loc'] = nn.Parameter(torch.zeros(normal_size),
                                                                        requires_grad=False)
            self.parameter_dict[dist]['Normal']['scale'] = nn.Parameter(torch.tensor(sigma_dict[dist], dtype=torch.float32),
                                                                        requires_grad=False)
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist]['Normal']['loc'],
                                                            scale=self.parameter_dict[dist]['Normal']['scale'])
                
    def _update_distributions(self):
        for dist in self.parameter_dict:
            self.distribution_dict[dist]['Delta'] = Delta(v=self.parameter_dict[dist]['Delta']['v'],
                                                          log_density=self.parameter_dict[dist]['Delta']['log_density'])
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist]['Normal']['loc'],
                                                            scale=self.parameter_dict[dist]['Normal']['scale'])
    
    def get_samples(self):
        result_dict = {"pi": None,
                       "z_mu": None,
                       "z_Sigma": None,
                       "w_mu": None}
        for dist in result_dict:
            sample1 = self.distribution_dict[dist]['Delta'].rsample()
            sample2 = self.distribution_dict[dist]['Normal'].rsample()
            result_dict[dist] =  torch.cat((sample1, sample2), dim=0)
        return result_dict
    
    def forward(self):
        self._update_distributions()
        return self.distribution_dict
    

class q_beta_class(nn.Module):
    def __init__(self,
                 y_dim: int, 
                 x_dim: int,
                 n_conditions: int,
                 n_cell_types: int) -> None:
        super().__init__()
        
        self.parameter_dict = {"pi": {"Delta": {},
                                      "Normal": {}},
                               "z_mu": {"Delta": {},
                                        "Normal": {}},
                               "z_Sigma": {"Delta": {},
                                           "Normal": {}},
                               "w_mu": {"Delta": {},
                                        "Normal": {}}}
        
        self.distribution_dict = {"pi": {},
                                  "z_mu": {},
                                  "z_Sigma": {},
                                  "w_mu": {}}
        
        for dist in self.parameter_dict:
            # The first distribution is Delta centered at 0 
            if dist == "pi":
                delta_size = (1, n_cell_types)
                normal_size = (n_conditions-1, n_cell_types)
            else:
                delta_size = (1, y_dim, x_dim)
                normal_size = (n_conditions-1, y_dim, x_dim)
            self.parameter_dict[dist]['Delta']['v'] = nn.Parameter(torch.zeros(delta_size),
                                                                    requires_grad=False)
            self.parameter_dict[dist]['Delta']['log_density'] = nn.Parameter(torch.ones(delta_size),
                                                                                requires_grad=False)
            self.distribution_dict[dist]['Delta'] = Delta(v=self.parameter_dict[dist]['Delta']['v'],
                                                          log_density=self.parameter_dict[dist]['Delta']['log_density'])
            self.parameter_dict[dist]['Normal']['loc'] = nn.Parameter(torch.randn(normal_size),
                                                                        requires_grad=True)
            self.parameter_dict[dist]['Normal']['scale'] = nn.Parameter(torch.randn(normal_size),
                                                                        requires_grad=True)
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist]['Normal']['loc'],
                                                            scale=F.softplus(self.parameter_dict[dist]['Normal']['scale']))
    
    def _update_distributions(self):
        for dist in self.parameter_dict:
            self.distribution_dict[dist]['Delta'] = Delta(v=self.parameter_dict[dist]['Delta']['v'],
                                                          log_density=self.parameter_dict[dist]['Delta']['log_density'])
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist]['Normal']['loc'],
                                                            scale=F.softplus(self.parameter_dict[dist]['Normal']['scale']))
    
    def get_samples(self):
        result_dict = {"pi": None,
                       "z_mu": None,
                       "z_Sigma": None,
                       "w_mu": None}
        for dist in result_dict:
            sample1 = self.distribution_dict[dist]['Delta'].rsample()
            sample2 = self.distribution_dict[dist]['Normal'].rsample()
            result_dict[dist] =  torch.cat((sample1, sample2), dim=0)      
        return result_dict
        
    def forward(self):
        self._update_distributions()
        return self.distribution_dict 

