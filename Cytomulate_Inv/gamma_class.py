import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal
from pyro.distributions import Delta

class p_log_var_class(nn.Module):
    def __init__(self,
                 mu_dict: dict={"pi": 0,
                                "z_mu": 0,
                                "z_Sigma": 0,
                                "w_mu": 0},
                 sigma_dict: dict={"pi": 0.1,
                                   "z_mu": 0.1,
                                   "z_Sigma": 0.1,
                                   "w_mu": 0.1}) -> None:
        super().__init__()
        
        self.parameter_dict = {"pi": {"Normal": {}},
                               "z_mu": {"Normal": {}},
                               "z_Sigma": {"Normal": {}},
                               "w_mu": {"Normal": {}}}
        
        self.distribution_dict = {"pi": {},
                                  "z_mu": {},
                                  "z_Sigma": {},
                                  "w_mu": {}}
        
        for dist in self.parameter_dict:
            self.parameter_dict[dist]['Normal']['loc'] = nn.Parameter(torch.tensor(mu_dict[dist], dtype=torch.float32),
                                                                      requires_grad=False)
            self.parameter_dict[dist]['Normal']['scale'] = nn.Parameter(torch.tensor(sigma_dict[dist], dtype=torch.float32),
                                                                        requires_grad=False)
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist]['Normal']['loc'],
                                                            scale=self.parameter_dict[dist]['Normal']['scale'])
        
    def _update_distributions(self):
        for dist in self.parameter_dict:
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist]['Normal']['loc'],
                                                            scale=self.parameter_dict[dist]['Normal']['scale']) 
    
    def get_samples(self):
        result_dict = {"pi": None,
                       "z_mu": None,
                       "z_Sigma": None,
                       "w_mu": None}
        for dist in result_dict:
            sample = self.distribution_dict[dist]['Normal'].rsample()
            result_dict[dist] =  sample  
        return result_dict
    
    def forward(self):
        self._update_distributions()
        return self.distribution_dict
    

class q_log_var_class(nn.Module):
    def __init__(self,
                 sigma_dict: dict={"pi": 0.1,
                                   "z_mu": 0.1,
                                   "z_Sigma": 0.1,
                                   "w_mu": 0.1}) -> None:
        super().__init__()

        self.parameter_dict = {"pi": {"Normal": {}},
                               "z_mu": {"Normal": {}},
                               "z_Sigma": {"Normal": {}},
                               "w_mu": {"Normal": {}}}
        
        self.distribution_dict = {"pi": {},
                                  "z_mu": {},
                                  "z_Sigma": {},
                                  "w_mu": {}}
        
        for dist in self.parameter_dict:
            self.parameter_dict[dist]['Normal']['scale'] = nn.Parameter(torch.tensor(sigma_dict[dist], dtype=torch.float32),
                                                                        requires_grad=False)
            self.parameter_dict[dist]['Normal']['loc'] = nn.Parameter(torch.randn(1), 
                                                                      requires_grad=True)
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist]['Normal']['loc'],
                                                            scale=self.parameter_dict[dist]['Normal']['scale'])
        
    
    def _update_distributions(self, 
                              gamma_samples: dict):
        for dist in self.parameter_dict:
            self.parameter_dict[dist]['Normal']['loc'] = torch.log(gamma_samples[dist].var()) - torch.pow(self.parameter_dict[dist]['Normal']['scale'],2)/2
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist]['Normal']['loc'],
                                                            scale=self.parameter_dict[dist]['Normal']['scale'])
    
    def get_samples(self):
        result_dict = {"pi": None,
                       "z_mu": None,
                       "z_Sigma": None,
                       "w_mu": None}
        for dist in result_dict:
            sample = self.distribution_dict[dist]['Normal'].rsample()
            result_dict[dist] =  sample
        return result_dict
    
    def forward(self, 
                gamma_samples: dict):
        self._update_distributions(gamma_samples=gamma_samples)
        return self.distribution_dict
        

class p_gamma_class(nn.Module):
    def __init__(self,
                 y_dim,
                 x_dim,
                 n_cell_types: int,
                 n_subjects: int) -> None:
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
                normal_size = (n_subjects-1, n_cell_types)
            else:
                delta_size = (1, y_dim, x_dim)
                normal_size = (n_subjects-1, y_dim, x_dim)

            self.parameter_dict[dist]['Delta']['v'] = nn.Parameter(torch.zeros(delta_size),
                                                                    requires_grad=False)
            self.parameter_dict[dist]['Delta']['log_density'] = nn.Parameter(torch.ones(delta_size),
                                                                                requires_grad=False)
            self.distribution_dict[dist]['Delta'] = Delta(v=self.parameter_dict[dist]['Delta']['v'],
                                                          log_density=self.parameter_dict[dist]['Delta']['log_density'])
            self.parameter_dict[dist]['Normal']['loc'] = nn.Parameter(torch.zeros(normal_size),
                                                                        requires_grad=False)
            self.parameter_dict[dist]['Normal']['scale'] = nn.Parameter(torch.tensor(1, dtype=torch.float32),
                                                                        requires_grad=False)
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist]['Normal']['loc'],
                                                            scale=self.parameter_dict[dist]['Normal']['scale'])
        
    def _update_distributions(self, 
                              log_var_samples: dict):
        for dist in self.parameter_dict:
            self.parameter_dict[dist]['Normal']['scale'] = torch.exp(log_var_samples[dist]/2)
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
        
    def forward(self, 
                log_var_samples: dict):
        self._update_distributions(log_var_samples=log_var_samples)
        return self.distribution_dict
    
    
class q_gamma_class(nn.Module):
    def __init__(self,
                 y_dim,
                 x_dim,
                 n_cell_types: int,
                 n_subjects: int) -> None:
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
            if dist == "pi":
                delta_size = (1, n_cell_types)
                normal_size = (n_subjects-1, n_cell_types)
            else:
                delta_size = (1, y_dim, x_dim)
                normal_size = (n_subjects-1, y_dim, x_dim)
                
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
