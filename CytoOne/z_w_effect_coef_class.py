import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from pyro.distributions import Delta 

from CytoOne.base_class import component_base_class


class p_effect_z_w_coef_class(component_base_class):
    def __init__(self,
                 y_dim: int,
                 n_batches: int,
                 n_conditions: int,
                 n_subjects: int,
                 n_cell_types: int): 
        super().__init__(stage_to_change="expression effect estimation",
                         distribution_info={"alpha_loc": (n_batches, y_dim),
                                            "beta_loc": (n_conditions, y_dim),
                                            "gamma_loc": (n_subjects, y_dim),
                                            "theta_loc": (n_cell_types, y_dim),
                                            "alpha_theta_loc": (n_batches*n_cell_types, y_dim),
                                            "beta_theta_loc": (n_conditions*n_cell_types, y_dim),
                                            "alpha_scale": (n_batches, y_dim),
                                            "beta_scale": (n_conditions, y_dim),
                                            "gamma_scale": (n_subjects, y_dim),
                                            "theta_scale": (n_cell_types, y_dim),
                                            "alpha_theta_scale": (n_batches*n_cell_types, y_dim),
                                            "beta_theta_scale": (n_conditions*n_cell_types, y_dim)})
        
        self.y_dim = y_dim
        self.n_batches = n_batches
        self.n_conditions = n_conditions
        self.n_subjects = n_subjects
        self.n_cell_types = n_cell_types
        
    def _update_distributions(self,
                              loc_eta_sample: torch.tensor,
                              scale_eta_sample: torch.tensor) -> None:
        
        gamma_loc_scale = torch.sqrt(torch.exp(loc_eta_sample))
        gamma_scale_scale = torch.sqrt(torch.exp(scale_eta_sample))
        
        for dist in self.distribution_dict:
            if (self.distribution_info_dict[dist][0] == 1) or \
               (("_theta_" in dist) and (self.distribution_info_dict[dist][0] == self.n_cell_types)) or \
               (("_theta_" in dist) and (self.n_cell_types == 1)):
                self.distribution_dict[dist] = Independent(Delta(v=torch.zeros(self.distribution_info_dict[dist]),
                                                                 log_density=torch.ones(self.distribution_info_dict[dist])),
                                                           reinterpreted_batch_ndims=len(self.distribution_info_dict[dist]))
            else:
                if dist == "gamma_loc":
                    self.distribution_dict[dist] = Independent(Normal(loc=torch.zeros(self.distribution_info_dict[dist]),
                                                                      scale=torch.ones(self.distribution_info_dict[dist])*gamma_loc_scale),
                                                               reinterpreted_batch_ndims=len(self.distribution_info_dict[dist])) 
                elif dist == "gamma_scale":
                    self.distribution_dict[dist] = Independent(Normal(loc=torch.zeros(self.distribution_info_dict[dist]),
                                                                      scale=torch.ones(self.distribution_info_dict[dist])*gamma_scale_scale),
                                                               reinterpreted_batch_ndims=len(self.distribution_info_dict[dist])) 
                else:
                    self.distribution_dict[dist] = Independent(Normal(loc=torch.zeros(self.distribution_info_dict[dist]),
                                                                      scale=torch.ones(self.distribution_info_dict[dist])),
                                                               reinterpreted_batch_ndims=len(self.distribution_info_dict[dist]))
        

class q_effect_z_w_coef_class(component_base_class):
    def __init__(self,
                 y_dim: int,
                 n_batches: int,
                 n_conditions: int,
                 n_subjects: int,
                 n_cell_types: int):
        super().__init__(stage_to_change = "expression effect estimation",
                         distribution_info={"alpha_loc": (n_batches, y_dim),
                                            "beta_loc": (n_conditions, y_dim),
                                            "gamma_loc": (n_subjects, y_dim),
                                            "theta_loc": (n_cell_types, y_dim),
                                            "alpha_theta_loc": (n_batches*n_cell_types, y_dim),
                                            "beta_theta_loc": (n_conditions*n_cell_types, y_dim),
                                            "alpha_scale": (n_batches, y_dim),
                                            "beta_scale": (n_conditions, y_dim),
                                            "gamma_scale": (n_subjects, y_dim),
                                            "theta_scale": (n_cell_types, y_dim),
                                            "alpha_theta_scale": (n_batches*n_cell_types, y_dim),
                                            "beta_theta_scale": (n_conditions*n_cell_types, y_dim)})
    
        self.y_dim = y_dim
        self.n_batches = n_batches
        self.n_conditions = n_conditions
        self.n_subjects = n_subjects
        self.n_cell_types = n_cell_types
        
        self.parameter_dict = nn.ParameterDict({})
        
        for dist in self.distribution_dict:
            if (self.distribution_info_dict[dist][0] > 1) and \
               (("_theta_" not in dist) or (self.distribution_info_dict[dist][0] != n_cell_types)) and \
               (("_theta_" not in dist) or (self.n_cell_types > 1)):
                self.parameter_dict.update({dist+"_loc": nn.Parameter(torch.randn(self.distribution_info_dict[dist]),
                                                                            requires_grad=True),
                                            dist+"_scale": nn.Parameter(torch.randn(self.distribution_info_dict[dist]),
                                                                            requires_grad=True)})
    
    def _update_distributions(self):
        for dist in self.distribution_dict:
            if (self.distribution_info_dict[dist][0] == 1) or \
               (("_theta_" in dist) and (self.distribution_info_dict[dist][0] == self.n_cell_types)) or \
               (("_theta_" in dist) and (self.n_cell_types == 1)):
                self.distribution_dict[dist] = Independent(Delta(v=torch.zeros(self.distribution_info_dict[dist]),
                                                                 log_density=torch.ones(self.distribution_info_dict[dist])),
                                                           reinterpreted_batch_ndims=len(self.distribution_info_dict[dist]))
            else:
                self.distribution_dict[dist] = Independent(Normal(loc=self.parameter_dict[dist+'_loc'],
                                                                  scale=F.softplus(self.parameter_dict[dist+'_scale'], beta=1)+0.00001),
                                                            reinterpreted_batch_ndims=len(self.distribution_info_dict[dist]))


class p_effect_z_w_hyper_coef_class(component_base_class):
    def __init__(self,
                 y_dim: int):
        super().__init__(stage_to_change = "expression effect estimation",
                         distribution_info={"loc_eta": (1, y_dim),
                                            "scale_eta": (1, y_dim)})
        
        self.y_dim = y_dim
        
    def _update_distributions(self):
        for dist in self.distribution_dict:
            self.distribution_dict[dist] = Independent(Normal(loc=torch.zeros(self.distribution_info_dict[dist]),
                                                              scale=torch.ones(self.distribution_info_dict[dist])),
                                                       reinterpreted_batch_ndims=2)
        
        
class q_effect_z_w_hyper_coef_class(component_base_class):
    def __init__(self,
                 y_dim: int,
                 n_subjects: int):
        super().__init__(stage_to_change = "expression effect estimation",
                         distribution_info={"loc_eta": (1, y_dim),
                                            "scale_eta": (1, y_dim)})
        
        self.y_dim = y_dim
        self.n_subjects = n_subjects
        self.parameter_dict = nn.ParameterDict({})
        
        for dist in self.distribution_dict:
            if n_subjects > 1:
                self.parameter_dict.update({dist+"_loc": nn.Parameter(torch.randn(self.distribution_info_dict[dist]),
                                                                            requires_grad=True),
                                            dist+"_scale": nn.Parameter(torch.randn(self.distribution_info_dict[dist]),
                                                                            requires_grad=True)})
        
    def _update_distributions(self):
        for dist in self.distribution_dict:
            if self.n_subjects == 1:
                self.distribution_dict[dist] = Independent(Normal(loc=torch.zeros(self.distribution_info_dict[dist]),
                                                                  scale=torch.ones(self.distribution_info_dict[dist])),
                                                           reinterpreted_batch_ndims=2)
            else:
                self.distribution_dict[dist] = Independent(Normal(loc=self.parameter_dict[dist+'_loc'],
                                                                  scale=F.softplus(self.parameter_dict[dist+'_scale'], beta=1)+0.00001),
                                                            reinterpreted_batch_ndims=len(self.dist_shape_dict[dist]))
    