import torch 
import torch.nn as nn 
from torch.distributions import Normal, Independent

from Cytomulate_Inv.basic_distributions import zero_inflated_lognormal,\
                                mollified_uniform
                                
from pyro.distributions import Delta 

from Cytomulate_Inv.network_modules.manifestation_prior_modules import w_prior_module,\
                                                                        z_prior_module
from Cytomulate_Inv.network_modules.manifestation_likelihood_modules import y_module
from Cytomulate_Inv.network_modules.manifestation_posterior_modules import w_posterior_module,\
                                                                            z_posterior_module

class manifestation_likelihood_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 distribution_type: str="ZILN",
                 mu_normal_scale: float=0.01) -> None:
        super().__init__()
        self.distribution_type = distribution_type
        self.y_module = y_module(y_dim=y_dim,
                                 distribution_type=distribution_type,
                                 mu_normal_scale=mu_normal_scale)
    
    def forward(self, w, z):
        y_dict = self.y_module(w=w, z=z)   
        if self.distribution_type == "ZILN":
            return {
                'likelihood': Independent(Delta(**y_dict),
                                          reinterpreted_batch_ndims=1)
            }
        if self.distribution_type == "N":
            return {
                'likelihood': Independent(Normal(**y_dict),
                                          reinterpreted_batch_ndims=1)
            }
        if self.distribution_type == "MU":
            return {
                'likelihood': Independent(mollified_uniform(**y_dict),
                                          reinterpreted_batch_ndims=1)
            }


class manifestation_prior_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int=2,
                 distribution_type: str="ZILN") -> None:
        super().__init__()
        
        self.distribution_type = distribution_type
        self.w_module = w_prior_module(y_dim=y_dim,
                                       x_dim=x_dim,
                                       distribution_type=distribution_type)
        self.z_module = z_prior_module(y_dim=y_dim,
                                       x_dim=x_dim,
                                       distribution_type=distribution_type)
        
    def forward(self, x):
        # The forward passes in priors only 
        # need to output distributions 
        w_dict = self.w_module(x=x)
        z_dict = self.z_module(x=x)
        if self.distribution_type == "ZILN":
            return {
                'w_prior': Independent(Delta(**w_dict),
                                       reinterpreted_batch_ndims=1),
                'z_prior': Independent(zero_inflated_lognormal(**z_dict),
                                       reinterpreted_batch_ndims=1)
            }
        if self.distribution_type in ["N", "MU"]:
            return {
                'w_prior': Independent(Normal(**w_dict),
                                       reinterpreted_batch_ndims=1),
                'z_prior': Independent(Normal(**z_dict),
                                       reinterpreted_batch_ndims=1)
            }


class manifestation_posterior_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 distribution_type: str="ZILN") -> None:
        super().__init__()
        self.distribution_type = distribution_type
        self.w_module = w_posterior_module(y_dim=y_dim,
                                            distribution_type=distribution_type)
        self.z_module = z_posterior_module(y_dim=y_dim,
                                            distribution_type=distribution_type)
        
        
    def forward(self, y):
        w_dict = self.w_module(y=y)
        z_dict = self.z_module(y=y)
        if self.distribution_type == "ZILN":
            return {
                'w_posterior': Independent(Delta(**w_dict),
                                       reinterpreted_batch_ndims=1),
                'z_posterior': Independent(Delta(**z_dict),
                                       reinterpreted_batch_ndims=1)
            }
        if self.distribution_type in ["N", "MU"]:
            return {
                'w_posterior': Independent(Normal(**w_dict),
                                       reinterpreted_batch_ndims=1),
                'z_posterior': Independent(Normal(**z_dict),
                                       reinterpreted_batch_ndims=1)
            }

        