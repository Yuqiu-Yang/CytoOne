import torch 
import torch.nn as nn 
from torch.distributions.kl import kl_divergence, register_kl

from Cytomulate_Inv.manifestation import manifestation_likelihood_class,\
                          manifestation_prior_class,\
                          manifestation_posterior_class

from Cytomulate_Inv.bottleneck import bottleneck_prior_class,\
                                      bottleneck_posterior_class      

from Cytomulate_Inv.typing import typing_prior_class, \
                                  typing_posterior_class                                      

from Cytomulate_Inv.utilities import *              
   
       
class cytof_model(nn.Module):
    def __init__(self,
                 y_dim: int, 
                 x_dim: int,
                 distribution_type: str,
                 mu_normal_scale: float,
                 x_scale: float,
                 n_cell_types: int,
                 beta: float) -> None:
        super().__init__()
        
        self.manifestation_likelihood = manifestation_likelihood_class(y_dim=y_dim,
                                                                       distribution_type=distribution_type,
                                                                       mu_normal_scale=mu_normal_scale)
        self.manifestation_prior = manifestation_prior_class(y_dim=y_dim,
                                                             x_dim=x_dim,
                                                             distribution_type=distribution_type)
        self.manifestation_posterior = manifestation_posterior_class(y_dim=y_dim,
                                                                     distribution_type=distribution_type)
        self.bottleneck_prior = bottleneck_prior_class(scale=x_scale)
        self.bottleneck_posterior = bottleneck_posterior_class(y_dim=y_dim,
                                                               x_dim=x_dim)
        self.typing_prior = typing_prior_class(n_cell_types=n_cell_types)
        self.typing_posterior = typing_posterior_class(x_dim=x_dim,
                                                       n_cell_types=n_cell_types,
                                                       beta=beta)
            
    def get_distributions(self, y):
        
        # To generate all the distributions, we start from
        # the posteriors 
        # The steps are almost always: generate the distributions
        # Then sample from the distributions 
        ####################################
        # POSTERIORS 
        ##############
        # w and z
        q_w_z_dict = self.manifestation_posterior(y=y)
        w_sample = q_w_z_dict['w_posterior'].rsample()
        z_sample = q_w_z_dict['z_posterior'].rsample()
        ###############
        # x
        q_x_dict = self.bottleneck_posterior(w=w_sample,
                                             z=z_sample)
        x_sample = q_x_dict['x_posterior'].rsample()
        ##############
        # Pi
        q_pi_dict = self.typing_posterior(x=x_sample)
        embedding = q_pi_dict['embedding']
        ####################################
        # PRIORS 
        #########
        # y
        p_y_dict = self.manifestation_likelihood(w=w_sample,
                                                 z=z_sample)
        log_likelihood = p_y_dict['likelihood'].log_prob(y)
        #########
        # w and z 
        p_w_z_dict = self.manifestation_prior(x=x_sample)
        #########
        # x 
        p_x_dict = self.bottleneck_prior(embedding=embedding)
        #########
        # Pi
        p_pi_dict = self.typing_prior()
        
        return {
            "p_pi_dict": p_pi_dict,
            "p_x_dict": p_x_dict,
            "p_w_z_dict": p_w_z_dict,
            "p_y_dict": p_y_dict,
            "log_likelihood": log_likelihood,
            "q_pi_dict": q_pi_dict,
            "q_x_dict": q_x_dict,
            "q_w_z_dict": q_w_z_dict
        }
    
    def forward(self, y):
        distribution_dict = self.get_distributions(y=y) 
        return distribution_dict
    
    def compute_loss(self, distribution_dict):
        reconstruction_error = distribution_dict['log_likelihood'].mean()
        
        kl_x = kl_divergence(distribution_dict['q_x_dict']['x_posterior'],
                             distribution_dict['p_x_dict']['x_prior']).mean()  
        kl_w = kl_divergence(distribution_dict['q_w_z_dict']['w_posterior'],
                             distribution_dict['p_w_z_dict']['w_prior']).mean()
        kl_z = kl_divergence(distribution_dict['q_w_z_dict']['z_posterior'],
                             distribution_dict['p_w_z_dict']['z_prior']).mean()
        kl_pi = kl_divergence(distribution_dict['q_pi_dict']['pi_posterior'],
                             distribution_dict['p_pi_dict']['pi_prior']).mean()
        
        elbo = reconstruction_error - kl_w - kl_z - kl_x - kl_pi
        
        return -elbo + distribution_dict['q_pi_dict']['vq_loss']