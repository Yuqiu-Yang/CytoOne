import torch 
from torch.distributions.kl import kl_divergence

from CytoOne.z_w_class import p_pretrain_z_class, p_pretrain_w_class, q_pretrain_z_class, q_pretrain_w_class
from CytoOne.y_class import p_y_class
from CytoOne.base_class import model_base_class


class pretrain_model(model_base_class):
    def __init__(self,
                 y_scale: float=0.001) -> None:
        super().__init__()
        self.p_z = p_pretrain_z_class()
        self.p_w = p_pretrain_w_class()
        self.p_y = p_y_class(y_scale=y_scale)
        self.q_z = q_pretrain_z_class()
        self.q_w = q_pretrain_w_class()
               
    def _update_distributions(self, 
                              y: torch.tensor):
        q_z_dict = self.q_z(y=y)
        z_samples = self.q_z.get_samples()
        
        q_w_dict = self.q_w(y=y)
        w_samples = self.q_w.get_samples()
        
        p_z_dict = self.p_z()
        p_w_dict = self.p_w()
        p_y_dict = self.p_y(z=z_samples['z'],
                            w=w_samples['w'])
        log_likelihood = p_y_dict['y'].log_prob(y)
        
        return {
            "q_z_dict": q_z_dict,
            "q_w_dict": q_w_dict,
            "p_z_dict": p_z_dict,
            "p_w_dict": p_w_dict,
            "p_y_dict": p_y_dict,
            "log_likelihood": log_likelihood
        } 
    
    def compute_loss(self, 
                     distribution_dict: dict):
        kl_z = kl_divergence(distribution_dict['q_z_dict']['z'], 
                             distribution_dict["p_z_dict"]['z']).mean()
        kl_w = kl_divergence(distribution_dict['q_w_dict']['w'],
                             distribution_dict["p_w_dict"]['w']).mean()
        reconstruction_error = distribution_dict['log_likelihood'].mean()

        elbo = reconstruction_error - kl_w - kl_z

        return -elbo
    