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
                 x_dim: int, 
                 y_scale: float,
                 n_batches: int,
                 n_conditions: int,
                 n_subjects: int,
                 n_cell_types: int,
                 model_check_point_path: Optional[str]=None,
                 model_device: Optional[Union[str, torch.device]]=None) -> None:
        super().__init__()
        
        if model_device is None:
            self.model_device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        elif isinstance(model_device, str):
            self.model_device = torch.device(model_device)
        else:
            self.model_device = model_device
        
        self.n_batches = n_batches
        self.n_conditions = n_conditions
        self.n_subjects = n_subjects
        self.n_cell_types = n_cell_types
        self.x_dim = x_dim
        self.current_stage = None
        
        self.p_y = p_y_class(model_device=self.model_device,
                             y_dim=y_dim,
                             y_scale=y_scale)
        self.p_z_w = p_z_w_class(model_device=self.model_device,
                                 y_dim=y_dim,
                                 x_dim=x_dim,
                                 n_batches=n_batches,
                                 n_conditions=n_conditions,
                                 n_subjects=n_subjects)
        self.q_z_w = q_z_w_class(model_device=self.model_device,
                                 y_dim=y_dim)
        if model_check_point_path is not None:
            self.q_z_w.load_pretrained_model(model_check_point_path=model_check_point_path)
            
        self.p_x = p_x_class(model_device=self.model_device,
                             x_dim=x_dim)
        self.q_x = q_x_class(model_device=self.model_device,
                             y_dim=y_dim,
                             x_dim=x_dim,
                             n_batches=n_batches,
                             n_conditions=n_conditions,
                             n_subjects=n_subjects)
                
        self.p_pi = p_pi_class(model_device=self.model_device,
                               n_cell_types=n_cell_types,
                               n_conditions=n_conditions,
                               n_subjects=n_subjects)
        self.q_pi = q_pi_class(model_device=self.model_device,
                               x_dim=x_dim,
                               n_cell_types=n_cell_types,
                               vq_vae_weight=0.25)
        
    def initialize_pi_distributions(self,
                                    n_cell_types: int):
        self.n_cell_types = n_cell_types
        self.p_pi = p_pi_class(model_device=self.model_device,
                               n_cell_types=n_cell_types,
                               n_conditions=self.n_conditions,
                               n_subjects=self.n_subjects)
        self.q_pi = q_pi_class(model_device=self.model_device,
                               x_dim=self.x_dim,
                               n_cell_types=n_cell_types,
                               vq_vae_weight=0.25)

    def get_posterior_samples(self,
                              y: torch.tensor,
                              FB: torch.tensor,
                              FC: torch.tensor,
                              RS: torch.tensor,
                              mode: str="validation",
                              get_mean: bool=False):
        assert mode in ["training", "validation"], "mode has to be either training or validation"
        if mode == "training":
            self.train()
        else:
            self.eval()
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
            z_w_samples = self.q_z_w.get_samples(get_mean=get_mean)
            # x 
            q_x_dict = self.q_x(z=z_w_samples['z'],
                                w=z_w_samples['w'],
                                FB=FB,
                                FC=FC,
                                RS=RS)
            x_samples = self.q_x.get_samples(get_mean=get_mean)

            q_pi_dict = {}
            if self.current_stage == "clustering":
                q_pi_dict = self.q_pi(x=x_samples['x']) 
        return q_z_w_dict, q_x_dict, q_pi_dict, z_w_samples, x_samples
    
    def get_latent_samples(self,
                           FC: torch.tensor,
                           RS: torch.tensor,
                           q_pi_dict: Optional[dict]=None,
                           mode: str="validation",
                           get_mean: bool=False):
        assert mode in ["training", "validation"], "mode has to be either training or validation"
        if mode == "training":
            self.train()
        else:
            self.eval()
        with torch.set_grad_enabled(mode=="training"):
            # x
            p_pi_dict = {}
            p_pi_samples = None
            p_x_samples = None
            if self.current_stage == "dimension reduction":
                p_x_dict = self.p_x()
                if q_pi_dict is None:
                    p_x_samples = self.p_x.get_samples(get_mean=get_mean)
            elif self.current_stage == "clustering":
                p_pi_dict = self.p_pi(FC=FC,
                                      RS=RS)
                if q_pi_dict is None:
                    p_pi_samples = self.p_pi.get_samples()
                    one_hot_encoding = torch.zeros(FC.size(0), 
                                                   self.q_pi.cell_embeddings.weight.size(0))
                    one_hot_encoding.scatter_(1, p_pi_samples['pi'], 1)
                    
                    embedding = torch.matmul(one_hot_encoding, self.q_pi.cell_embeddings.weight)
                    p_x_dict = self.p_x(embedding=embedding)
                    p_x_samples = self.p_x.get_samples(get_mean=get_mean)
                else:
                    p_x_dict = self.p_x(embedding=q_pi_dict['embedding'])
        
        return p_pi_dict, p_x_dict, p_pi_samples, p_x_samples
    
    def reconstruct_y(self,
                      FB: torch.tensor,
                      FC: torch.tensor,
                      RS: torch.tensor,
                      x_samples: dict, 
                      z_w_samples: Optional[dict]=None, 
                      y: Optional[torch.tensor]=None,
                      mode: str="validation",
                      get_mean: bool=False):
        assert mode in ["training", "validation"], "mode has to be either training or validation"
        if mode == "training":
            self.train()
        else:
            self.eval()
        with torch.set_grad_enabled(mode=="training"):
            # z and w
            p_z_w_dict = self.p_z_w(x=x_samples['x'],
                                    FB=FB,
                                    FC=FC,
                                    RS=RS)
            log_likelihood = 0
            if z_w_samples is None:
                p_z_w_samples = self.p_z_w.get_samples(get_mean=get_mean)
                # y
                p_y_dict = self.p_y(z=p_z_w_samples['z'],
                                    w=p_z_w_samples['w']) 
            else:
                p_z_w_samples = None
                # y
                p_y_dict = self.p_y(z=z_w_samples['z'],
                                    w=z_w_samples['w']) 
                if y is not None:
                    log_likelihood = p_y_dict['y'].log_prob(y)
                
            y_samples = self.p_y.get_samples(get_mean=get_mean)
        return p_z_w_dict, p_y_dict, p_z_w_samples, y_samples, log_likelihood
    
    def _update_distributions(self, 
                              y: torch.tensor,
                              FB: torch.tensor,
                              FC: torch.tensor,
                              RS: torch.tensor):
        q_z_w_dict, q_x_dict, q_pi_dict, z_w_samples, x_samples = self.get_posterior_samples(y=y,
                                                                                             FB=FB,
                                                                                             FC=FC,
                                                                                             RS=RS,
                                                                                             mode="training",
                                                                                             get_mean=False)
        p_pi_dict, p_x_dict, _, _ = self.get_latent_samples(FC=FC,
                                                            RS=RS,
                                                            q_pi_dict=q_pi_dict,
                                                            mode="training",
                                                            get_mean=False)
        p_z_w_dict, p_y_dict, _, _, log_likelihood = self.reconstruct_y(FB=FB,
                                                                        FC=FC,
                                                                        RS=RS,
                                                                        x_samples=x_samples,
                                                                        z_w_samples=z_w_samples,
                                                                        y=y,
                                                                        mode='training',
                                                                        get_mean=False)
            
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
    
    def normalize_samples(self,
                          y: torch.tensor,
                          FB: torch.tensor,
                          FC: torch.tensor,
                          RS: torch.tensor,
                          normalize_to_batch: Optional[int]=0,
                          normalize_to_condition: Optional[int]=0,
                          normalize_to_subject: Optional[int]=0,
                          get_mean: bool=False):
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
            _, _, _, _, x_samples = self.get_posterior_samples(y=y,
                                                               FB=FB,
                                                               FC=FC,
                                                               RS=RS,
                                                               mode="validation",
                                                               get_mean=get_mean)
            _, _, _, y_samples, _ = self.reconstruct_y(FB=nFB,
                                                       FC=nFC,
                                                       RS=nRS,
                                                       x_samples=x_samples,
                                                       z_w_samples=None,
                                                       y=None,
                                                       mode='validation',
                                                       get_mean=get_mean)
        return y_samples
            
    def compute_loss(self, 
                     distribution_dict: dict,
                     show_details: bool=False):
        reconstruction_error = distribution_dict['log_likelihood'].mean()
        
        kl_x = kl_divergence(distribution_dict['q_x_dict']['x'],
                            distribution_dict['p_x_dict']['x']).mean()  
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
        if show_details:
            print("="*25)
            print("likelihood is {}, kl_x is {}, kl_z is {}, kl_w is {}, kl_pi is {} vq_loss is {}".format(reconstruction_error,
                                                                                                           kl_x,
                                                                                                           kl_z,
                                                                                                           kl_w,
                                                                                                           kl_pi,
                                                                                                           vq_loss))
            print("="*25)
        return -elbo + vq_loss