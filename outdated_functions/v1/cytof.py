import torch 
import torch.nn as nn 
from CytoOne.utilities import _kl_delta_delta, _kl_delta_ziln, _kl_delta_categorical            
from torch.distributions.kl import kl_divergence

from CytoOne.y_class import p_y_class
from CytoOne.z_w_class import p_z_w_class, q_z_w_class
from CytoOne.x_class import p_x_class, q_x_class
from CytoOne.pi_class import p_pi_class, q_pi_class
from CytoOne.alpha_class import p_alpha_class, q_alpha_class
from CytoOne.beta_class import p_beta_class, q_beta_class
from CytoOne.gamma_class import p_log_var_class, p_gamma_class, q_log_var_class, q_gamma_class
                
from typing import Optional 
                  

class cytof_model(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int, 
                 distribution_type: str,
                 fixed_noise_level: bool,
                 n_batches: int,
                 n_conditions: int,
                 n_subjects: int,
                 n_cell_types: int,
                 sigma_y_dict: dict,
                 sigma_x_dict: dict,
                 sigma_alpha_dict: dict,
                 sigma_beta_dict: dict,
                 mu_log_var_dict: dict,
                 sigma_log_var_dict: dict,
                 vq_vae_weight: float,
                 global_weight: float,
                 local_weight: float) -> None:
        super().__init__()
        
        self.distribution_type = distribution_type
        self.n_batches = n_batches
        self.n_conditions = n_conditions
        self.n_subjects = n_subjects
        self.global_weight = global_weight
        self.local_weight = local_weight
        
        self.p_y = p_y_class(distribution_type=distribution_type,
                             sigma_dict=sigma_y_dict,
                             fixed_noise_level=fixed_noise_level)
        self.p_z_w = p_z_w_class(y_dim=y_dim,
                                 x_dim=x_dim,
                                 distribution_type=distribution_type)
        self.q_z_w = q_z_w_class(y_dim=y_dim,
                                 distribution_type=distribution_type)
        self.p_x = p_x_class(sigma_dict=sigma_x_dict)
        self.q_x = q_x_class(y_dim=y_dim,
                             x_dim=x_dim,
                             sigma_dict=sigma_x_dict,
                             n_batches=n_batches,
                             n_conditions=n_conditions,
                             n_subjects=n_subjects)        
        self.p_pi = p_pi_class(n_cell_types=n_cell_types)
        self.q_pi = q_pi_class(x_dim=x_dim,
                               n_cell_types=n_cell_types,
                               vq_vae_weight=vq_vae_weight)
        self.p_alpha = p_alpha_class(y_dim=y_dim,
                                     x_dim=x_dim,
                                     n_batches=n_batches,
                                     sigma_dict=sigma_alpha_dict)
        self.q_alpha = q_alpha_class(y_dim=y_dim,
                                     x_dim=x_dim,
                                     n_batches=n_batches)
        self.p_beta = p_beta_class(y_dim=y_dim,
                                   x_dim=x_dim,
                                   n_conditions=n_conditions,
                                   n_cell_types=n_cell_types,
                                   sigma_dict=sigma_beta_dict)
        self.q_beta = q_beta_class(y_dim=y_dim,
                                   x_dim=x_dim,
                                   n_conditions=n_conditions,
                                   n_cell_types=n_cell_types)
        self.p_log_var = p_log_var_class(mu_dict=mu_log_var_dict,
                                         sigma_dict=sigma_log_var_dict)
        self.q_log_var = q_log_var_class(n_subjects=n_subjects,
                                         mu_dict=mu_log_var_dict,
                                         sigma_dict=sigma_log_var_dict)
        self.p_gamma = p_gamma_class(y_dim=y_dim,
                                     x_dim=x_dim,
                                     n_cell_types=n_cell_types,
                                     n_subjects=n_subjects)
        self.q_gamma = q_gamma_class(y_dim=y_dim,
                                     x_dim=x_dim,
                                     n_cell_types=n_cell_types,
                                     n_subjects=n_subjects)
            
    def _update_distributions(self, 
                              y: torch.tensor,
                              FB: torch.tensor,
                              FC: torch.tensor,
                              RS: torch.tensor,
                              sigma_y: Optional[float]=None,
                              mode="training"):
        with torch.set_grad_enabled(mode=="training"):
            # To generate all the distributions, we start from
            # the posteriors 
            # The steps are almost always: generate the distributions
            # Then sample from the distributions 
            ####################################
            # Q 
            ##############
            # alpha
            q_alpha_dict = self.q_alpha()
            alpha_samples = self.q_alpha.get_samples()
            # beta
            q_beta_dict = self.q_beta()
            beta_samples = self.q_beta.get_samples()
            # gamma 
            q_gamma_dict = self.q_gamma()
            gamma_samples = self.q_gamma.get_samples()
            # log_var
            q_log_var_dict = self.q_log_var(gamma_samples=gamma_samples)
            log_var_samples = self.q_log_var.get_samples()
            # z and w
            q_z_w_dict = self.q_z_w(y=y)
            z_w_samples = self.q_z_w.get_samples()
            # x 
            q_x_dict = self.q_x(z=z_w_samples['z'],
                                w=z_w_samples['w'],
                                alpha_z_mu_sample=alpha_samples['z_mu'],
                                beta_z_mu_sample=beta_samples['z_mu'],
                                gamma_z_mu_sample=gamma_samples['z_mu'],
                                alpha_z_Sigma_sample=alpha_samples['z_Sigma'],
                                beta_z_Sigma_sample=beta_samples['z_Sigma'],
                                gamma_z_Sigma_sample=gamma_samples['z_Sigma'],
                                alpha_w_mu_sample=alpha_samples['w_mu'],
                                beta_w_mu_sample=beta_samples['w_mu'],
                                gamma_w_mu_sample=gamma_samples['w_mu'],
                                FB=FB,
                                FC=FC,
                                RS=RS)
            x_samples = self.q_x.get_sample()
            # Pi
            q_pi_dict = self.q_pi(x=x_samples['x'])
            ####################################
            # P
            ##############
            # log_var
            p_log_var_dict = self.p_log_var()
            # alpha 
            p_alpha_dict = self.p_alpha()
            # beta
            p_beta_dict = self.p_beta()
            # gamma
            p_gamma_dict = self.p_gamma(log_var_samples=log_var_samples)
            # Pi
            p_pi_dict = self.p_pi(beta_pi_sample=beta_samples['pi'],
                                gamma_pi_sample=gamma_samples['pi'],
                                FC=FC,
                                RS=RS)
            # x
            p_x_dict = self.p_x(embedding=q_pi_dict['embedding'])
            # z and w
            p_z_w_dict = self.p_z_w(x=x_samples['x'],
                                    alpha_z_mu_sample=alpha_samples['z_mu'],
                                    beta_z_mu_sample=beta_samples['z_mu'],
                                    gamma_z_mu_sample=gamma_samples['z_mu'],
                                    alpha_z_Sigma_sample=alpha_samples['z_Sigma'],
                                    beta_z_Sigma_sample=beta_samples['z_Sigma'],
                                    gamma_z_Sigma_sample=gamma_samples['z_Sigma'],
                                    alpha_w_mu_sample=alpha_samples['w_mu'],
                                    beta_w_mu_sample=beta_samples['w_mu'],
                                    gamma_w_mu_sample=gamma_samples['w_mu'],
                                    FB=FB,
                                    FC=FC,
                                    RS=RS)
            # y
            if self.distribution_type in ["N", "MU"]:
                p_y_dict = self.p_y(z=z_w_samples['z'],
                                    w=z_w_samples['w'],
                                    sigma_y=sigma_y)
            else:
                p_y_dict = {"y": p_z_w_dict['z']}
            log_likelihood = p_y_dict['y'].log_prob(y)
            
            
            return {
                "q_log_var_dict": q_log_var_dict,
                "p_log_var_dict": p_log_var_dict,
                "q_alpha_dict": q_alpha_dict,
                "p_alpha_dict": p_alpha_dict,
                "q_beta_dict": q_beta_dict,
                "p_beta_dict": p_beta_dict,
                "q_gamma_dict": q_gamma_dict,
                "p_gamma_dict": p_gamma_dict,
                "q_pi_dict": q_pi_dict,
                "p_pi_dict": p_pi_dict,
                "q_x_dict": q_x_dict,
                "p_x_dict": p_x_dict,
                "q_z_w_dict": q_z_w_dict,
                "p_z_w_dict": p_z_w_dict,
                "p_y_dict": p_y_dict,
                "log_likelihood": log_likelihood
            }
    
    def forward(self,
                y: torch.tensor,
                FB: torch.tensor,
                FC: torch.tensor,
                RS: torch.tensor,
                sigma_y: Optional[float]=None):
        distribution_dict = self._update_distributions(y=y,
                                                       FB=FB,
                                                       FC=FC,
                                                       RS=RS,
                                                       sigma_y=sigma_y)
        return distribution_dict
    
    def compute_loss(self, distribution_dict):
        reconstruction_error = distribution_dict['log_likelihood'].mean()
        
        # log_var 
        kl_log_var_gamma_pi = kl_divergence(distribution_dict['q_log_var_dict']['pi'],
                                            distribution_dict['p_log_var_dict']['pi'])
        kl_log_var_gamma_z_mu = kl_divergence(distribution_dict['q_log_var_dict']['z_mu'],
                                              distribution_dict['p_log_var_dict']['z_mu'])
        kl_log_var_gamma_z_Sigma = kl_divergence(distribution_dict['q_log_var_dict']['z_Sigma'],
                                                 distribution_dict['p_log_var_dict']['z_Sigma'])
        kl_log_var_gamma_w_mu = kl_divergence(distribution_dict['q_log_var_dict']['w_mu'],
                                              distribution_dict['p_log_var_dict']['w_mu'])
        
        # alpha 
        kl_alpha_z_mu = kl_divergence(distribution_dict['q_alpha_dict']['z_mu']['Normal'],
                                      distribution_dict['p_alpha_dict']['z_mu']['Normal']).sum().nan_to_num_(nan=0.0)
        kl_alpha_z_Sigma = kl_divergence(distribution_dict['q_alpha_dict']['z_Sigma']['Normal'],
                                         distribution_dict['p_alpha_dict']['z_Sigma']['Normal']).sum().nan_to_num_(nan=0.0)
        kl_alpha_w_mu = kl_divergence(distribution_dict['q_alpha_dict']['w_mu']['Normal'],
                                      distribution_dict['p_alpha_dict']['w_mu']['Normal']).sum().nan_to_num_(nan=0.0)
        # beta 
        kl_beta_pi = kl_divergence(distribution_dict["q_beta_dict"]['pi']['Normal'],
                                   distribution_dict['p_beta_dict']['pi']['Normal']).sum().nan_to_num_(nan=0.0)
        kl_beta_z_mu = kl_divergence(distribution_dict['q_beta_dict']['z_mu']['Normal'],
                                      distribution_dict['p_beta_dict']['z_mu']['Normal']).sum().nan_to_num_(nan=0.0)
        kl_beta_z_Sigma = kl_divergence(distribution_dict['q_beta_dict']['z_Sigma']['Normal'],
                                         distribution_dict['p_beta_dict']['z_Sigma']['Normal']).sum().nan_to_num_(nan=0.0)
        kl_beta_w_mu = kl_divergence(distribution_dict['q_beta_dict']['w_mu']['Normal'],
                                      distribution_dict['p_beta_dict']['w_mu']['Normal']).sum().nan_to_num_(nan=0.0)
        # gamma 
        kl_gamma_pi = kl_divergence(distribution_dict["q_gamma_dict"]['pi']['Normal'],
                                    distribution_dict['p_gamma_dict']['pi']['Normal']).sum().nan_to_num_(nan=0.0)
        kl_gamma_z_mu = kl_divergence(distribution_dict['q_gamma_dict']['z_mu']['Normal'],
                                      distribution_dict['p_gamma_dict']['z_mu']['Normal']).sum().nan_to_num_(nan=0.0)
        kl_gamma_z_Sigma = kl_divergence(distribution_dict['q_gamma_dict']['z_Sigma']['Normal'],
                                         distribution_dict['p_gamma_dict']['z_Sigma']['Normal']).sum().nan_to_num_(nan=0.0)
        kl_gamma_w_mu = kl_divergence(distribution_dict['q_gamma_dict']['w_mu']['Normal'],
                                      distribution_dict['p_gamma_dict']['w_mu']['Normal']).sum().nan_to_num_(nan=0.0)
        # local variables 
        kl_pi = kl_divergence(distribution_dict['q_pi_dict']['pi'],
                              distribution_dict['p_pi_dict']['pi']).mean()
        kl_x = kl_divergence(distribution_dict['q_x_dict']['x'],
                             distribution_dict['p_x_dict']['x']).mean()  
        kl_z = kl_divergence(distribution_dict['q_z_w_dict']['z'],
                             distribution_dict['p_z_w_dict']['z']).mean()
        kl_w = kl_divergence(distribution_dict['q_z_w_dict']['w'],
                             distribution_dict['p_z_w_dict']['w']).mean()
        
        global_kl = - kl_log_var_gamma_pi - kl_log_var_gamma_z_mu - kl_log_var_gamma_z_Sigma - kl_log_var_gamma_w_mu - \
                kl_alpha_z_mu - kl_alpha_z_Sigma - kl_alpha_w_mu - \
                kl_beta_pi - kl_beta_z_mu - kl_beta_z_Sigma - kl_beta_w_mu - \
                kl_gamma_pi - kl_gamma_z_mu - kl_gamma_z_Sigma - kl_gamma_w_mu
        
        local_kl = - kl_pi - kl_x - kl_z - kl_w 
        
        elbo = reconstruction_error + self.global_weight * global_kl + self.local_weight * local_kl
        
        return -elbo + self.local_weight * distribution_dict['q_pi_dict']['vq_loss']

    def get_samples(self,
                    y: torch.tensor,
                    FB: torch.tensor,
                    FC: torch.tensor,
                    RS: torch.tensor):
        self.eval()
        with torch.no_grad():
            # To generate all the distributions, we start from
            # the posteriors 
            # The steps are almost always: generate the distributions
            # Then sample from the distributions 
            ####################################
            # Q 
            ##############
            # alpha
            q_alpha_dict = self.q_alpha()
            alpha_samples = self.q_alpha.get_samples()
            # beta
            q_beta_dict = self.q_beta()
            beta_samples = self.q_beta.get_samples()
            # gamma 
            q_gamma_dict = self.q_gamma()
            gamma_samples = self.q_gamma.get_samples()
            # log_var
            q_log_var_dict = self.q_log_var(gamma_samples=gamma_samples)
            log_var_samples = self.q_log_var.get_samples()
            # z and w
            q_z_w_dict = self.q_z_w(y=y)
            z_w_samples = self.q_z_w.get_samples()
            # x 
            q_x_dict = self.q_x(z=z_w_samples['z'],
                                w=z_w_samples['w'],
                                alpha_z_mu_sample=alpha_samples['z_mu'],
                                beta_z_mu_sample=beta_samples['z_mu'],
                                gamma_z_mu_sample=gamma_samples['z_mu'],
                                alpha_z_Sigma_sample=alpha_samples['z_Sigma'],
                                beta_z_Sigma_sample=beta_samples['z_Sigma'],
                                gamma_z_Sigma_sample=gamma_samples['z_Sigma'],
                                alpha_w_mu_sample=alpha_samples['w_mu'],
                                beta_w_mu_sample=beta_samples['w_mu'],
                                gamma_w_mu_sample=gamma_samples['w_mu'],
                                FB=FB,
                                FC=FC,
                                RS=RS)
            x_samples = self.q_x.get_sample()
            # Pi
            q_pi_dict = self.q_pi(x=x_samples['x'])
            
            return {
                "alpha_samples": alpha_samples,
                "beta_samples": beta_samples,
                "gamma_samples": gamma_samples,
                "log_var_samples": log_var_samples,
                "z_w_samples": z_w_samples,
                "x_samples": x_samples,
                "embedding": q_pi_dict['embedding']
            }    