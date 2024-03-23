import torch 
import torch.nn as nn 
import torch.nn.functional as F
from CytoOne.utilities import _kl_delta_delta, _kl_delta_ziln, _kl_delta_categorical            
from torch.distributions.kl import kl_divergence

from CytoOne.y_class import p_y_class
from CytoOne.z_w_class import p_z_w_class, q_z_w_class
from CytoOne.x_class import p_x_class, q_x_class
from CytoOne.pi_class import p_pi_class, q_pi_class

from CytoOne.base_class import model_base_class

from typing import Optional, Union


class cytoone_model(model_base_class):
    def __init__(self,
                 y_dim: int,
                 x_dim: Union[int, list], 
                 n_batches: int,
                 n_conditions: int,
                 n_subjects: int,
                 n_cell_types: int,
                 model_check_point_path: Optional[str]=None) -> None:
        super().__init__()
        
        self.n_batches = n_batches
        self.n_conditions = n_conditions
        self.n_subjects = n_subjects
        self.n_cell_types = n_cell_types
        if isinstance(x_dim, int):
            x_dim = [x_dim]
        self.x_dim = x_dim
        self.current_stage = None
        
        self.p_y = p_y_class(y_dim=y_dim)
        self.p_z_w = p_z_w_class(y_dim=y_dim,
                                 x_dim=x_dim[-1],
                                 n_batches=n_batches,
                                 n_conditions=n_conditions,
                                 n_subjects=n_subjects)
        self.q_z_w = q_z_w_class(y_dim=y_dim)
        if model_check_point_path is not None:
            self.q_z_w.load_pretrained_model(model_check_point_path=model_check_point_path)
            
        self.p_x = p_x_class(x_dim=x_dim)
        self.q_x = q_x_class(y_dim=y_dim,
                             x_dim=x_dim,
                             n_batches=n_batches,
                             n_conditions=n_conditions,
                             n_subjects=n_subjects)
                
        # self.p_x_vq_vae = p_x_vq_vae_class(x_dim=x_dim)
        self.p_pi = p_pi_class(n_cell_types=n_cell_types,
                               n_conditions=n_conditions,
                               n_subjects=n_subjects)
        self.q_pi = q_pi_class(x_dim=x_dim[0],
                               n_cell_types=n_cell_types,
                               vq_vae_weight=0.25)
        
    def initialize_pi_distributions(self,
                                    n_cell_types: int):
        self.n_cell_types = n_cell_types
        self.p_pi = p_pi_class(n_cell_types=n_cell_types,
                               n_conditions=self.n_conditions)
        self.q_pi = q_pi_class(x_dim=self.x_dim[0],
                               n_cell_types=n_cell_types,
                               vq_vae_weight=0.25)
    
    def _update_distributions(self, 
                              y: torch.tensor,
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
            ##############
            # z and w
            q_z_w_dict = self.q_z_w(y=y)
            z_w_samples = self.q_z_w.get_samples()
            # x 
            q_x_dict = self.q_x(z=z_w_samples['z'],
                                w=z_w_samples['w'],
                                FB=FB,
                                FC=FC,
                                RS=RS)
            x_samples = self.q_x.get_samples()

            q_pi_dict = {}
            if self.current_stage == "clustering":
                q_pi_dict = self.q_pi(x=x_samples['x'+str(self.q_x.T)])
                 
            ####################################
            # P
            ##############
            # x
            p_pi_dict = {}
            if self.current_stage == "dimension reduction":
                p_x_dict = self.p_x(q_x_samples=x_samples)
            elif self.current_stage == "clustering":
                p_pi_dict = self.p_pi(FC=FC,
                                      RS=RS)
                p_x_dict = self.p_x(embedding=q_pi_dict['embedding'],
                                    q_x_samples=x_samples)
            # z and w
            p_z_w_dict = self.p_z_w(x=x_samples['x0'],
                                    FB=FB,
                                    FC=FC,
                                    RS=RS)
            
            # y
            p_y_dict = self.p_y(z=z_w_samples['z'],
                                w=z_w_samples['w'])
            log_likelihood = p_y_dict['y'].log_prob(y)
            
            return {
                "q_pi_dict": q_pi_dict,
                "p_pi_dict": p_pi_dict,
                "q_x_dict": q_x_dict,
                "p_x_dict": p_x_dict,
                "q_z_w_dict": q_z_w_dict,
                "p_z_w_dict": p_z_w_dict,
                "p_y_dict": p_y_dict,
                "log_likelihood": log_likelihood
            }

    def get_posterior_samples(self,
                              y: torch.tensor,
                              FB: torch.tensor,
                              FC: torch.tensor,
                              RS: torch.tensor,
                              get_mean: bool=False):
        self.eval()
        with torch.no_grad():
            # z and w
            q_z_w_dict = self.q_z_w(y=y)
            z_w_samples = self.q_z_w.get_samples(get_mean=get_mean)
            # x 
            q_x_dict = self.q_x(z=z_w_samples['z'],
                                w=z_w_samples['w'],
                                FB=FB,
                                FC=FC,
                                RS=RS)
            x_samples = self.q_x.get_samples(get_mean=get_mean)
            q_pi_dict = self.q_pi(x=x_samples['x'+str(self.q_x.T)])
            
        return z_w_samples['z'], z_w_samples['w'], x_samples['x'+str(self.q_x.T)], q_pi_dict['embedding'], q_pi_dict['one_hot_encoding']
    
    def normalize_samples(self,
                          y: torch.tensor,
                          FB: torch.tensor,
                          FC: torch.tensor,
                          RS: torch.tensor,
                          normalize_to_batch: Optional[int]=0,
                          normalize_to_condition: Optional[int]=0,
                          normalize_to_subject: Optional[int]=0):
        if normalize_to_batch is None:
            nFB = FB
        else:
            nFB = torch.zeros_like(FB)
            nFB[:, normalize_to_batch] = 1
        if normalize_to_condition is None:
            nFC = FC
        else:
            nFC = torch.zeros_like(FC)
            nFC[:, normalize_to_condition] = 1
        if normalize_to_subject is None:
            nRS = RS
        else:
            nRS = torch.zeros_like(RS)
            nRS[:, normalize_to_subject] = 1
        
        self.eval()
        with torch.no_grad():
            # z and w
            q_z_w_dict = self.q_z_w(y=y)
            z_w_samples = self.q_z_w.get_samples()
            # x 
            q_x_dict = self.q_x(z=z_w_samples['z'],
                                w=z_w_samples['w'],
                                FB=FB,
                                FC=FC,
                                RS=RS)
            x_samples = self.q_x.get_samples()
            q_pi_dict = self.q_pi(x=x_samples['x'])
            
            
            p_z_w_dict = self.p_z_w(x=x_samples['x'],
                                    FB=nFB,
                                    FC=nFC,
                                    RS=nRS)
            p_z_w_samples = self.p_z_w.get_samples()
            
            w = torch.round(F.sigmoid(p_z_w_samples['w']))
    
            z = torch.exp(p_z_w_samples['z'])
            y = w * z 
        return y
            
    def generate_new_samples(self,
                             n_samples):
        self.eval()
        with torch.no_grad():
            pass  
            
    def compute_loss(self, 
                     distribution_dict: dict):
        reconstruction_error = distribution_dict['log_likelihood'].mean()
        
        kl_x = 0
        for t in range(self.p_x.T + 1):
            kl_x += kl_divergence(distribution_dict['q_x_dict']['x'+str(t)],
                                 distribution_dict['p_x_dict']['x'+str(t)]).mean()  
        kl_z = kl_divergence(distribution_dict['q_z_w_dict']['z'],
                            distribution_dict['p_z_w_dict']['z']).mean()
        kl_w = kl_divergence(distribution_dict['q_z_w_dict']['w'],
                            distribution_dict['p_z_w_dict']['w']).mean()
        kl_pi = 0
        vq_loss = 0
        if self.current_stage == "clustering":
            kl_pi = kl_divergence(distribution_dict['q_pi_dict']['pi'],
                                  distribution_dict['p_pi_dict']['pi']).mean()
            vq_loss = distribution_dict['q_pi_dict']['vq_loss']
        local_kl = - kl_x - kl_z - kl_w - kl_pi

        elbo = reconstruction_error + local_kl
        
        return -elbo + vq_loss