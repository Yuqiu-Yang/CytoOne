# PyTorch 
import torch 
import torch.nn as nn 
from torch.distributions.kl import kl_divergence

from CytoOne.pi_class import p_effect_pi_class
from CytoOne.pi_effect_coef_class import p_effect_pi_coef_class, p_effect_pi_hyper_coef_class, \
                                         q_effect_pi_coef_class, q_effect_pi_hyper_coef_class

from CytoOne.base_class import model_base_class


class abundance_model(model_base_class):
    def __init__(self,
                 n_conditions: int,
                 n_subjects: int,
                 n_cell_types: int):
        super().__init__()
        
        assert (n_cell_types > 1), "The number of cell types has to be > 1. Otherwise, there is nothing to estimate."
        assert (n_conditions > 1) or (n_subjects > 1), "You need to have at least 2 conditions or 2 subjects to estimate some effects."

        self.n_conditions = n_conditions
        self.n_subjects = n_subjects
        self.n_cell_types = n_cell_types
        
        self.p_pi = p_effect_pi_class()
        self.p_pi_coef = p_effect_pi_coef_class(n_conditions=n_conditions,
                                                n_subjects=n_subjects,
                                                n_cell_types=n_cell_types)
        self.p_pi_hyper_coef = p_effect_pi_hyper_coef_class(n_subjects=n_subjects,
                                                            n_cell_types=n_cell_types)
        self.q_pi_coef = q_effect_pi_coef_class(n_conditions=n_conditions,
                                                n_subjects=n_subjects,
                                                n_cell_types=n_cell_types)
        self.q_pi_hyper_coef = q_effect_pi_hyper_coef_class(n_subjects=n_subjects,
                                                            n_cell_types=n_cell_types)
        
    def _update_distributions(self, 
                              one_hot_index: torch.tensor,
                              FC: torch.tensor,
                              RS: torch.tensor, 
                              mode="training"):
        with torch.set_grad_enabled(mode=="training"):
            # To generate all the distributions, we start from
            # the posteriors 
            # The steps are almost always: generate the distributions
            # Then sample from the distributions 
            ####################################
            # Q
            ################
            q_pi_coef_dict = self.q_pi_coef()
            pi_coef_samples = self.q_pi_coef.get_samples()
            
            q_pi_hyper_coef_dict = self.q_pi_hyper_coef()
            pi_hyper_coef_samples = self.q_pi_hyper_coef.get_samples()
                 
            ####################################
            # P
            ##############
            p_pi_hyper_coef_dict = self.p_pi_hyper_coef()
            p_pi_coef_dict = self.p_pi_coef(eta_sample=pi_hyper_coef_samples['eta'])
            
            index = torch.argmax(one_hot_index, dim=1)
            p_pi_dict = self.p_pi(FC=FC,
                                  RS=RS,
                                  beta_sample=pi_coef_samples['beta'],
                                  gamma_sample=pi_coef_samples['gamma'])
            log_likelihood = p_pi_dict['pi'].log_prob(index)
            
            return {
                "q_pi_hyper_coef_dict": q_pi_hyper_coef_dict,
                "q_pi_coef_dict": q_pi_coef_dict,
                "p_pi_hyper_coef_dict": p_pi_hyper_coef_dict,
                "p_pi_coef_dict": p_pi_coef_dict,
                "p_pi_dict": p_pi_dict,
                "log_likelihood": log_likelihood
            }
    
    def compute_loss(self, distribution_dict):
        reconstruction_error = distribution_dict['log_likelihood'].sum()

        kl_hyper_coef = kl_divergence(distribution_dict['q_pi_hyper_coef_dict']['eta'],
                                      distribution_dict['p_pi_hyper_coef_dict']['eta']).sum()
        kl_coef = 0 
        for dist in distribution_dict['q_pi_coef_dict']:
            kl_coef += kl_divergence(distribution_dict['q_pi_coef_dict'][dist],
                                     distribution_dict['p_pi_coef_dict'][dist]).sum()  
        
        elbo = reconstruction_error - kl_coef - kl_hyper_coef
        
        return -elbo
