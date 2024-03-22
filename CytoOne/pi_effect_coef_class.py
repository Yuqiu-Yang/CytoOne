# PyTorch 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from pyro.distributions import Delta 

from CytoOne.base_class import component_base_class


class p_effect_pi_coef_class(component_base_class):
    def __init__(self,
                 model_device: torch.device,
                 n_conditions: int,
                 n_subjects: int,
                 n_cell_types: int): 
        super().__init__(stage_to_change="abundance effect estimation",
                         distribution_info={"beta": (n_conditions, n_cell_types),
                                            "gamma": (n_subjects, n_cell_types)})
        self.model_device = model_device
        
        self.n_conditions = n_conditions
        self.n_subjects = n_subjects
        self.n_cell_types = n_cell_types
    
    def _update_distributions(self,
                              eta_sample: torch.tensor) -> None:
        gamma_scale = torch.sqrt(torch.exp(eta_sample))
        for dist in self.distribution_dict:
            if (self.distribution_info_dict[dist][0] == 1):
                self.distribution_dict[dist] = Independent(Delta(v=torch.zeros(self.distribution_info_dict[dist],
                                                                               device=self.model_device),
                                                                 log_density=torch.ones(self.distribution_info_dict[dist],
                                                                                        device=self.model_device)),
                                                           reinterpreted_batch_ndims=2) 
            else:
                if dist == "gamma":
                    self.distribution_dict[dist] = Independent(Normal(loc=torch.zeros(self.distribution_info_dict[dist],
                                                                                      device=self.model_device),
                                                                    scale=torch.ones(self.distribution_info_dict[dist],
                                                                                     device=self.model_device)*gamma_scale),
                                                            reinterpreted_batch_ndims=2) 
                else:
                    self.distribution_dict[dist] = Independent(Normal(loc=torch.zeros(self.distribution_info_dict[dist],
                                                                                      device=self.model_device),
                                                                    scale=torch.ones(self.distribution_info_dict[dist],
                                                                                     device=self.model_device)),
                                                            reinterpreted_batch_ndims=2)
    

class q_effect_pi_coef_class(component_base_class):
    def __init__(self,
                 model_device: torch.device,
                 n_conditions: int,
                 n_subjects: int,
                 n_cell_types: int):
        super().__init__(stage_to_change="abundance effect estimation",
                         distribution_info={"beta": (n_conditions, n_cell_types),
                                            "gamma": (n_subjects, n_cell_types)})
        self.model_device = model_device
        self.n_conditions = n_conditions
        self.n_subjects = n_subjects
        self.n_cell_types = n_cell_types
        
        self.parameter_dict = nn.ParameterDict({})

        for dist in self.distribution_dict:
            if (self.distribution_info_dict[dist][0] > 1):
                self.parameter_dict.update({dist+"_loc": nn.Parameter(torch.randn(self.distribution_info_dict[dist]),
                                                                            requires_grad=True),
                                            dist+"_scale": nn.Parameter(torch.randn(self.distribution_info_dict[dist]),
                                                                            requires_grad=True)})
        
    def _update_distributions(self):
        for dist in self.distribution_dict:
            if (self.distribution_info_dict[dist][0] == 1):
                self.distribution_dict[dist] = Independent(Delta(v=torch.zeros(self.distribution_info_dict[dist],
                                                                               device=self.model_device),
                                                                 log_density=torch.ones(self.distribution_info_dict[dist],
                                                                                        device=self.model_device)),
                                                           reinterpreted_batch_ndims=2)
            else:
                self.distribution_dict[dist] = Independent(Normal(loc=self.parameter_dict[dist+'_loc'],
                                                                  scale=F.softplus(self.parameter_dict[dist+'_scale'], beta=1)+0.00001),
                                                            reinterpreted_batch_ndims=2)


class p_effect_pi_hyper_coef_class(component_base_class):
    def __init__(self,
                 model_device: torch.device,
                 n_subjects: int,
                 n_cell_types: int):
        super().__init__(stage_to_change="abundance effect estimation",
                         distribution_info={"eta": (1, n_cell_types)})
        self.model_device = model_device
        self.n_subjects = n_subjects
        self.n_cell_types = n_cell_types
        
    def _update_distributions(self):
        for dist in self.distribution_dict:
            self.distribution_dict[dist] = Independent(Normal(loc=torch.zeros((1, self.n_cell_types),
                                                                              device=self.model_device),
                                                              scale=torch.ones((1, self.n_cell_types),
                                                                               device=self.model_device)),
                                                       reinterpreted_batch_ndims=2)  
        
        
class q_effect_pi_hyper_coef_class(component_base_class):
    def __init__(self,
                 model_device: torch.device,
                 n_subjects: int,
                 n_cell_types: int):
        super().__init__(stage_to_change="abundance effect estimation",
                         distribution_info={"eta": (1, n_cell_types)})       
        self.model_device = model_device
        self.n_subjects = n_subjects
        self.n_cell_types = n_cell_types
        
        self.parameter_dict = nn.ParameterDict({})
        if self.n_subjects > 1:
            for dist in self.distribution_dict:
                self.parameter_dict.update({dist+"_loc": nn.Parameter(torch.randn(self.distribution_info_dict[dist]),
                                                                            requires_grad=True),
                                            dist+"_scale": nn.Parameter(torch.randn(self.distribution_info_dict[dist]),
                                                                            requires_grad=True)})
    
    def _update_distributions(self):
        if self.n_subjects > 1:
            for dist in self.distribution_dict:
                self.distribution_dict[dist] = Independent(Normal(loc=self.parameter_dict[dist+'_loc'],
                                                                  scale=F.softplus(self.parameter_dict[dist+'_scale'], beta=1)+0.00001),
                                                        reinterpreted_batch_ndims=2)
        else:
            for dist in self.distribution_dict:
                self.distribution_dict[dist] = Independent(Normal(loc=torch.zeros(self.distribution_info_dict[dist],
                                                                                  device=self.model_device),
                                                                scale=torch.ones(self.distribution_info_dict[dist],
                                                                                 device=self.model_device)),
                                                        reinterpreted_batch_ndims=2)  
