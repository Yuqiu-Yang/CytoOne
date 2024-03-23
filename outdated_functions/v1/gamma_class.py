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
        
        # self.parameter_dict = {"pi": {},
        #                        "z_mu": {},
        #                        "z_Sigma": {},
        #                        "w_mu": {}}
        self.parameter_dict = nn.ParameterDict({})
        self.distribution_dict = {"pi": {},
                                  "z_mu": {},
                                  "z_Sigma": {},
                                  "w_mu": {}}
        
        for dist in self.distribution_dict:
            self.parameter_dict.update({dist+"_loc": nn.Parameter(torch.tensor(mu_dict[dist], dtype=torch.float32),
                                                            requires_grad=False),
                                        dist+"_scale": nn.Parameter(torch.tensor(sigma_dict[dist], dtype=torch.float32),
                                                              requires_grad=False)})
            self.distribution_dict[dist] = Normal(loc=self.parameter_dict[dist+'_loc'],
                                                  scale=self.parameter_dict[dist+'_scale'])
        
    def _update_distributions(self):
        for dist in self.distribution_dict:
            self.distribution_dict[dist] = Normal(loc=self.parameter_dict[dist+'_loc'],
                                                  scale=self.parameter_dict[dist+'_scale'])
    
    def get_samples(self):
        result_dict = {"pi": None,
                       "z_mu": None,
                       "z_Sigma": None,
                       "w_mu": None}
        for dist in result_dict:
            sample = self.distribution_dict[dist].rsample()
            result_dict[dist] =  sample  
        return result_dict
    
    def forward(self):
        self._update_distributions()
        return self.distribution_dict
    

class q_log_var_class(nn.Module):
    def __init__(self,
                 n_subjects: int,
                 mu_dict: dict={"pi": 0,
                                "z_mu": 0,
                                "z_Sigma": 0,
                                "w_mu": 0},
                 sigma_dict: dict={"pi": 0.1,
                                   "z_mu": 0.1,
                                   "z_Sigma": 0.1,
                                   "w_mu": 0.1}) -> None:
        super().__init__()

        # self.parameter_dict = {"pi": {},
        #                        "z_mu": {},
        #                        "z_Sigma": {},
        #                        "w_mu": {}}
        self.parameter_dict = nn.ParameterDict({})
        self.distribution_dict = {"pi": {},
                                  "z_mu": {},
                                  "z_Sigma": {},
                                  "w_mu": {}}
        
        for dist in self.distribution_dict:
            if n_subjects > 1.5:
                self.parameter_dict.update({dist+"_loc": nn.Parameter(torch.randn(1), 
                                                                    requires_grad=True),
                                            dist+"_scale": nn.Parameter(torch.tensor(sigma_dict[dist], dtype=torch.float32),
                                                                requires_grad=False)})
            else:
                self.parameter_dict.update({dist+"_loc": nn.Parameter(torch.tensor(mu_dict[dist], dtype=torch.float32), 
                                                                    requires_grad=False),
                                            dist+"_scale": nn.Parameter(torch.tensor(sigma_dict[dist], dtype=torch.float32),
                                                                requires_grad=False)})
            
            self.distribution_dict[dist] = Normal(loc=self.parameter_dict[dist+'_loc'],
                                                  scale=self.parameter_dict[dist+'_scale'])
        
    
    def _update_distributions(self, 
                              gamma_samples: dict):
        for dist in self.distribution_dict:
            if gamma_samples[dist].shape[0] > 1.5:
                self.parameter_dict[dist+'_loc'] = torch.log(gamma_samples[dist][1:,:].var()+0.00001) - torch.pow(self.parameter_dict[dist+'_scale'],2)/2
            self.distribution_dict[dist] = Normal(loc=self.parameter_dict[dist+'_loc'],
                                                  scale=self.parameter_dict[dist+'_scale'])
    
    def get_samples(self):
        result_dict = {"pi": None,
                       "z_mu": None,
                       "z_Sigma": None,
                       "w_mu": None}
        for dist in result_dict:
            sample = self.distribution_dict[dist].rsample()
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
        
        # self.parameter_dict = {"pi": {"Delta": {},
        #                               "Normal": {}},
        #                        "z_mu": {"Delta": {},
        #                                 "Normal": {}},
        #                        "z_Sigma": {"Delta": {},
        #                                    "Normal": {}},
        #                        "w_mu": {"Delta": {},
        #                                 "Normal": {}}}
        self.parameter_dict = nn.ParameterDict({})
        self.distribution_dict = {"pi": {},
                                  "z_mu": {},
                                  "z_Sigma": {},
                                  "w_mu": {}}
        
        for dist in self.distribution_dict:
            # The first distribution is Delta centered at 0 
            if dist == "pi":
                delta_size = (1, n_cell_types)
                normal_size = (n_subjects-1, n_cell_types)
            else:
                delta_size = (1, y_dim, x_dim)
                normal_size = (n_subjects-1, y_dim, x_dim)

            self.parameter_dict.update({dist+'_Delta_v': nn.Parameter(torch.zeros(delta_size),
                                                                      requires_grad=False),
                                        dist+"_Delta_log_density": nn.Parameter(torch.ones(delta_size),
                                                                                requires_grad=False),
                                        dist+"_Normal_loc": nn.Parameter(torch.zeros(normal_size),
                                                                         requires_grad=False),
                                        dist+"_Normal_scale": nn.Parameter(torch.tensor(1, dtype=torch.float32),
                                                                           requires_grad=False)})
            self.distribution_dict[dist]['Delta'] = Delta(v=self.parameter_dict[dist+'_Delta_v'],
                                                          log_density=self.parameter_dict[dist+'_Delta_log_density'])
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist+'_Normal_loc'],
                                                            scale=self.parameter_dict[dist+'_Normal_scale'])
        
    def _update_distributions(self, 
                              log_var_samples: dict):
        for dist in self.distribution_dict:
            self.parameter_dict[dist+'_Normal_scale'] = torch.exp(log_var_samples[dist]/2)
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist+'_Normal_loc'],
                                                            scale=self.parameter_dict[dist+'_Normal_scale'])
    
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
        
        # self.parameter_dict = {"pi": {"Delta": {},
        #                               "Normal": {}},
        #                        "z_mu": {"Delta": {},
        #                                 "Normal": {}},
        #                        "z_Sigma": {"Delta": {},
        #                                    "Normal": {}},
        #                        "w_mu": {"Delta": {},
        #                                 "Normal": {}}}
        self.parameter_dict = nn.ParameterDict({})
        self.distribution_dict = {"pi": {},
                                  "z_mu": {},
                                  "z_Sigma": {},
                                  "w_mu": {}}
        
        for dist in self.distribution_dict:
            if dist == "pi":
                delta_size = (1, n_cell_types)
                normal_size = (n_subjects-1, n_cell_types)
            else:
                delta_size = (1, y_dim, x_dim)
                normal_size = (n_subjects-1, y_dim, x_dim)
            
            self.parameter_dict.update({dist+'_Delta_v': nn.Parameter(torch.zeros(delta_size),
                                                                      requires_grad=False),
                                        dist+"_Delta_log_density": nn.Parameter(torch.ones(delta_size),
                                                                                requires_grad=False),
                                        dist+"_Normal_loc": nn.Parameter(torch.randn(normal_size),
                                                                         requires_grad=True),
                                        dist+"_Normal_scale": nn.Parameter(torch.randn(normal_size),
                                                                         requires_grad=True)})
            self.distribution_dict[dist]['Delta'] = Delta(v=self.parameter_dict[dist+'_Delta_v'],
                                                          log_density=self.parameter_dict[dist+'_Delta_log_density'])
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist+'_Normal_loc'],
                                                            scale=F.softplus(self.parameter_dict[dist+'_Normal_scale'], beta=1)+0.00001)
            
    def _update_distributions(self):
        for dist in self.distribution_dict:
            self.distribution_dict[dist]['Delta'] = Delta(v=self.parameter_dict[dist+'_Delta_v'],
                                                          log_density=self.parameter_dict[dist+'_Delta_log_density'])
            self.distribution_dict[dist]['Normal'] = Normal(loc=self.parameter_dict[dist+'_Normal_loc'],
                                                            scale=F.softplus(self.parameter_dict[dist+'_Normal_scale'], beta=1)+0.00001)
            
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
