import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from collections import OrderedDict

from CytoOne.basic_distributions import zero_inflated_lognormal
                                
from pyro.distributions import Delta 


class p_z_w_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int,
                 distribution_type: str="ZILN") -> None:
        super().__init__()
        assert distribution_type in ["ZILN", "N", "MU"], "distribution_type has to be one of ZILN, N, or MU"
        self.distribution_type = distribution_type
        
        self.mu_z_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=x_dim,
                                out_features=128,
                                bias=True)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=128,
                                out_features=256,
                                bias=True)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=256,
                                out_features=512,
                                bias=True)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=512,
                                out_features=y_dim,
                                bias=True))
        ]))
        
        self.log_Sigma_z_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=x_dim,
                                out_features=128,
                                bias=True)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=128,
                                out_features=256,
                                bias=True)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=256,
                                out_features=512,
                                bias=True)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=512,
                                out_features=y_dim,
                                bias=True))
        ]))
        
        self.mu_w_mapping = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(in_features=x_dim,
                                  out_features=128,
                                  bias=True)),
                ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
                ('fc2', nn.Linear(in_features=128,
                                  out_features=256,
                                  bias=True)),
                ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
                ('fc3', nn.Linear(in_features=256,
                                  out_features=512,
                                  bias=True)),
                ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
                ('fc4', nn.Linear(in_features=512,
                                  out_features=y_dim,
                                  bias=True))
        ]))
        
        if distribution_type == "ZILN":
            self.log_Sigma_w_mapping = nn.Linear(in_features=x_dim,
                                                 out_features=y_dim)
            self.log_Sigma_w_mapping.weight.data.zero_()
            self.log_Sigma_w_mapping.bias.data.fill_(1.0)
            for param in self.log_Sigma_w_mapping.parameters():
                param.requires_grad = False 
            
        else:
            self.log_Sigma_w_mapping = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(in_features=x_dim,
                                  out_features=128,
                                  bias=True)),
                ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
                ('fc2', nn.Linear(in_features=128,
                                  out_features=256,
                                  bias=True)),
                ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
                ('fc3', nn.Linear(in_features=256,
                                  out_features=512,
                                  bias=True)),
                ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
                ('fc4', nn.Linear(in_features=512,
                                  out_features=y_dim,
                                  bias=True))
            ]))
        
        self.distribution_dict = {"z": None,
                                  "w": None}
        
    def _update_distributions(self,
                              x: torch.tensor,
                              alpha_z_mu_sample: torch.tensor,
                              beta_z_mu_sample: torch.tensor,
                              gamma_z_mu_sample: torch.tensor,
                              alpha_z_Sigma_sample: torch.tensor,
                              beta_z_Sigma_sample: torch.tensor,
                              gamma_z_Sigma_sample: torch.tensor,
                              alpha_w_mu_sample: torch.tensor,
                              beta_w_mu_sample: torch.tensor,
                              gamma_w_mu_sample: torch.tensor,
                              FB: torch.tensor,
                              FC: torch.tensor,
                              RS: torch.tensor):
        if self.distribution_type == "ZILN":
            loc = self.mu_z_mapping(x) + \
                  torch.einsum("nb, bmd, nd -> nm", FB, alpha_z_mu_sample, x) + \
                  torch.einsum("nc, cmd, nd -> nm", FC, beta_z_mu_sample, x) + \
                  torch.einsum("ns, smd, nd -> nm", RS, gamma_z_mu_sample, x)
            scale = F.softplus(self.log_Sigma_z_mapping(x) + \
                               torch.einsum("nb, bmd, nd -> nm", FB, alpha_z_Sigma_sample, x) + \
                               torch.einsum("nc, cmd, nd -> nm", FC, beta_z_Sigma_sample, x) + \
                               torch.einsum("ns, smd, nd -> nm", RS, gamma_z_Sigma_sample, x), beta=1) + 0.00001
            probs = F.sigmoid(self.mu_w_mapping(x) + \
                              torch.einsum("nb, bmd, nd -> nm", FB, alpha_w_mu_sample, x) + \
                              torch.einsum("nc, cmd, nd -> nm", FC, beta_w_mu_sample, x) + \
                              torch.einsum("ns, smd, nd -> nm", RS, gamma_w_mu_sample, x))
            self.distribution_dict['z'] = Independent(zero_inflated_lognormal(loc=loc,
                                                                              scale=scale,
                                                                              gate=probs),
                                                      reinterpreted_batch_ndims=1)
            self.distribution_dict['w'] = Independent(Delta(v=self.log_Sigma_w_mapping(x),
                                                            log_density=1),
                                                      reinterpreted_batch_ndims=1) 
        else:
            z_loc = self.mu_z_mapping(x) + \
                    torch.einsum("nb, bmd, nd -> nm", FB, alpha_z_mu_sample, x) + \
                    torch.einsum("nc, cmd, nd -> nm", FC, beta_z_mu_sample, x) + \
                    torch.einsum("ns, smd, nd -> nm", RS, gamma_z_mu_sample, x)
            z_scale = F.softplus(self.log_Sigma_z_mapping(x) + \
                                 torch.einsum("nb, bmd, nd -> nm", FB, alpha_z_Sigma_sample, x) + \
                                 torch.einsum("nc, cmd, nd -> nm", FC, beta_z_Sigma_sample, x) + \
                                 torch.einsum("ns, smd, nd -> nm", RS, gamma_z_Sigma_sample, x), beta=1) + 0.00001
            w_loc = F.sigmoid(self.mu_w_mapping(x) + \
                              torch.einsum("nb, bmd, nd -> nm", FB, alpha_w_mu_sample, x) + \
                              torch.einsum("nc, cmd, nd -> nm", FC, beta_w_mu_sample, x) + \
                              torch.einsum("ns, smd, nd -> nm", RS, gamma_w_mu_sample, x))  
            w_scale = F.softplus(self.log_Sigma_w_mapping(x), beta=1) + 0.00001
            self.distribution_dict['z'] = Independent(Normal(loc=z_loc,
                                                             scale=z_scale),
                                                      reinterpreted_batch_ndims=1)
            self.distribution_dict['w'] = Independent(Normal(loc=w_loc,
                                                             scale=w_scale),
                                                      reinterpreted_batch_ndims=1)
            
    def get_samples(self):
        result_dict = {"z": None,
                       "w": None}
        for dist in self.distribution_dict:
            result_dict[dist] = self.distribution_dict[dist].rsample()
        return result_dict
    
    def forward(self,
                x: torch.tensor,
                alpha_z_mu_sample: torch.tensor,
                beta_z_mu_sample: torch.tensor,
                gamma_z_mu_sample: torch.tensor,
                alpha_z_Sigma_sample: torch.tensor,
                beta_z_Sigma_sample: torch.tensor,
                gamma_z_Sigma_sample: torch.tensor,
                alpha_w_mu_sample: torch.tensor,
                beta_w_mu_sample: torch.tensor,
                gamma_w_mu_sample: torch.tensor,
                FB: torch.tensor,
                FC: torch.tensor,
                RS: torch.tensor):
        self._update_distributions(x=x,
                                   alpha_z_mu_sample=alpha_z_mu_sample,
                                   beta_z_mu_sample=beta_z_mu_sample,
                                   gamma_z_mu_sample=gamma_z_mu_sample,
                                   alpha_z_Sigma_sample=alpha_z_Sigma_sample,
                                   beta_z_Sigma_sample=beta_z_Sigma_sample,
                                   gamma_z_Sigma_sample=gamma_z_Sigma_sample,
                                   alpha_w_mu_sample=alpha_w_mu_sample,
                                   beta_w_mu_sample=beta_w_mu_sample,
                                   gamma_w_mu_sample=gamma_w_mu_sample,
                                   FB=FB,
                                   FC=FC,
                                   RS=RS) 
        return self.distribution_dict
    

class q_z_w_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 distribution_type: str="ZILN") -> None:
        super().__init__()
        assert distribution_type in ["ZILN", "N", "MU"], "distribution_type has to be one of ZILN, N, or MU"
        self.distribution_type = distribution_type
        
        self.one_mapping = nn.Linear(in_features=y_dim,
                                     out_features=y_dim)
        self.one_mapping.weight.data.zero_()
        self.one_mapping.bias.data.fill_(1.0)
        for param in self.one_mapping.parameters():
            param.requires_grad = False 
        
        if distribution_type in ["N", "MU"]:
            self.u_z_mapping = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(in_features=y_dim,
                                        out_features=y_dim))
            ]))
            
            # self.u_z_mapping = nn.Parameter(torch.randn(1),
            #                                 requires_grad=True)
            
            self.log_s_z_mapping = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(in_features=y_dim,
                                        out_features=y_dim))
            ]))
            
            self.u_w_mapping = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(in_features=y_dim,
                                        out_features=y_dim))
            ]))
            
            # self.u_w_mapping = nn.Parameter(torch.randn(1),
            #                                 requires_grad=True)
            
            self.log_s_w_mapping = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(in_features=y_dim,
                                        out_features=y_dim))
            ]))
        else:
            self.u_z_mapping = nn.Sequential(OrderedDict([
                    ('fc1', nn.Identity())
            ]))
            self.log_s_z_mapping = self.one_mapping
            self.u_w_mapping = self.one_mapping
            self.log_s_w_mapping = self.one_mapping 
        
        self.distribution_dict = {"z": None,
                                  "w": None}
        
    def _update_distributions(self,
                              y: torch.tensor):
        if self.distribution_type == "ZILN":
            z_v = self.u_z_mapping(y)
            z_log_density = self.log_s_z_mapping(y)
            w_v = self.u_w_mapping(y)
            w_log_density = self.log_s_w_mapping(y)
            self.distribution_dict['z'] = Independent(Delta(v=z_v,
                                                            log_density=z_log_density),
                                                      reinterpreted_batch_ndims=1)
            self.distribution_dict['w'] = Independent(Delta(v=w_v,
                                                            log_density=w_log_density),
                                                      reinterpreted_batch_ndims=1)
        else:
            transformd_y = torch.asinh(y)
            z_loc = self.u_z_mapping(transformd_y)
            z_scale = F.softplus(self.log_s_z_mapping(transformd_y), beta=10) + 0.00001
            w_loc = self.u_w_mapping(transformd_y)
            w_scale = F.softplus(self.log_s_w_mapping(transformd_y), beta=10) + 0.00001
            self.distribution_dict['z'] = Independent(Normal(loc=z_loc,
                                                             scale=z_scale),
                                                      reinterpreted_batch_ndims=1)
            self.distribution_dict['w'] = Independent(Normal(loc=w_loc,
                                                             scale=w_scale),
                                                      reinterpreted_batch_ndims=1)  
    
    def get_samples(self):
        result_dict = {"z": None,
                       "w": None}
        for dist in self.distribution_dict:
            result_dict[dist] = self.distribution_dict[dist].rsample()
        return result_dict 
    
    def forward(self,
                y: torch.tensor):
        self._update_distributions(y=y)
        return self.distribution_dict 

