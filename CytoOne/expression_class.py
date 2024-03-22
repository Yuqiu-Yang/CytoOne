# PyTorch 
import torch 
from torch.distributions.kl import kl_divergence

from CytoOne.z_w_class import p_effect_z_w_class
from CytoOne.z_w_effect_coef_class import p_effect_z_w_coef_class, p_effect_z_w_hyper_coef_class,\
                                         q_effect_z_w_coef_class, q_effect_z_w_hyper_coef_class

from CytoOne.base_class import model_base_class

from typing import Optional, Union

class expression_model(model_base_class):
    def __init__(self,
                 y_dim: int,
                 n_batches: int,
                 n_conditions: int,
                 n_subjects: int,
                 n_cell_types: int,
                 effect_type: str,
                 model_device: Optional[Union[str, torch.device]]=None):
        super().__init__()
        
        assert (n_cell_types > 1) or (n_batches > 1) or (n_conditions > 1) or (n_subjects > 1), "You need to have at least 2 cell types or 2 batches or 2 conditions or 2 subjects to estimate some effects."
        assert effect_type in ["expression", "inflation"], "Illegal type..."
        
        if model_device is None:
            self.model_device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        elif isinstance(model_device, str):
            self.model_device = torch.device(model_device)
        else:
            self.model_device = model_device
        
        self.y_dim = y_dim
        self.n_batches = n_batches
        self.n_conditions = n_conditions
        self.n_subjects = n_subjects
        self.n_cell_types = n_cell_types
        self.effect_type = effect_type
        
        self.p_z_w = p_effect_z_w_class(model_device=self.model_device)
        self.p_z_w_coef = p_effect_z_w_coef_class(model_device=self.model_device,
                                                  y_dim=y_dim,
                                                  n_batches=n_batches,
                                                  n_conditions=n_conditions,
                                                  n_subjects=n_subjects,
                                                  n_cell_types=n_cell_types)
        self.q_z_w_coef = q_effect_z_w_coef_class(model_device=self.model_device,
                                                  y_dim=y_dim,
                                                  n_batches=n_batches,
                                                  n_conditions=n_conditions,
                                                  n_subjects=n_subjects,
                                                  n_cell_types=n_cell_types)
        self.p_z_w_hyper_coef = p_effect_z_w_hyper_coef_class(model_device=self.model_device,
                                                              y_dim=y_dim)
        self.q_z_w_hyper_coef = q_effect_z_w_hyper_coef_class(model_device=self.model_device,
                                                              y_dim=y_dim,
                                                              n_subjects=n_subjects)
        
        
    def _update_distributions(self, 
                              z_w: torch.tensor,
                              one_hot_index: torch.tensor,
                              FB: torch.tensor,
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
            q_z_w_coef_dict = self.q_z_w_coef()
            z_w_coef_samples = self.q_z_w_coef.get_samples()

            q_z_w_hyper_coef_dict = self.q_z_w_hyper_coef()
            z_w_hyper_coef_samples = self.q_z_w_hyper_coef.get_samples()
            
            ####################################
            # P
            ##############
            p_z_w_hyper_coef_dict = self.p_z_w_hyper_coef()
            
            p_z_w_coef_dict = self.p_z_w_coef(loc_eta_sample=z_w_hyper_coef_samples['loc_eta'],
                                              scale_eta_sample=z_w_hyper_coef_samples['scale_eta'])
            
            p_z_w_dict = self.p_z_w(one_hot_index=one_hot_index,
                                    FB=FB,
                                    FC=FC,
                                    RS=RS,
                                    theta_loc_sample=z_w_coef_samples['theta_loc'],
                                    alpha_loc_sample=z_w_coef_samples['alpha_loc'],
                                    beta_loc_sample=z_w_coef_samples['beta_loc'],
                                    gamma_loc_sample=z_w_coef_samples['gamma_loc'],
                                    alpha_theta_loc_sample=z_w_coef_samples['alpha_theta_loc'],
                                    beta_theta_loc_sample=z_w_coef_samples['beta_theta_loc'],
                                    theta_scale_sample=z_w_coef_samples['theta_scale'],
                                    alpha_scale_sample=z_w_coef_samples['alpha_scale'],
                                    beta_scale_sample=z_w_coef_samples['beta_scale'],
                                    gamma_scale_sample=z_w_coef_samples['gamma_scale'],
                                    alpha_theta_scale_sample=z_w_coef_samples['alpha_theta_scale'],
                                    beta_theta_scale_sample=z_w_coef_samples['beta_theta_scale'])
            log_likelihood = p_z_w_dict['z_w'].log_prob(z_w)
            
            return {
                "q_z_w_hyper_coef_dict": q_z_w_hyper_coef_dict,
                "p_z_w_hyper_coef_dict": p_z_w_hyper_coef_dict,
                "q_z_w_coef_dict": q_z_w_coef_dict,
                "p_z_w_coef_dict": p_z_w_coef_dict,
                "p_z_w_dict": p_z_w_dict,
                "log_likelihood": log_likelihood
            }
    
    def compute_loss(self,
                     distribution_dict: dict,
                     show_details: bool=False):
        reconstruction_error = distribution_dict['log_likelihood'].sum()
        
        kl_hyper_coef = 0
        for dist in distribution_dict['q_z_w_hyper_coef_dict']:
            kl_hyper_coef += kl_divergence(distribution_dict['q_z_w_hyper_coef_dict'][dist],
                                           distribution_dict['p_z_w_hyper_coef_dict'][dist]).sum()
        
        kl_coef = 0
        for dist in distribution_dict['q_z_w_coef_dict']:
            kl_coef += kl_divergence(distribution_dict['q_z_w_coef_dict'][dist],
                                    distribution_dict['p_z_w_coef_dict'][dist]).sum()
        
        elbo = reconstruction_error - kl_coef - kl_hyper_coef
        if show_details:
            print("="*25)
            print("likelihood is {}, kl_coef is {}, kl_hyper_coef is {}".format(reconstruction_error,
                                                                                kl_coef,
                                                                                kl_hyper_coef))
            print("="*25)
        return -elbo

