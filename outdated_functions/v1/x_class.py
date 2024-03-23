import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from collections import OrderedDict


class p_x_class(nn.Module):
    def __init__(self,
                 sigma_dict: dict={"x": 0.1}) -> None:
        super().__init__()
        
        self.scale = torch.tensor(sigma_dict['x'], dtype=torch.float32)
        
        self.distribution_dict = {"x": None}
    
    def _update_distribution(self, embedding):
        self.distribution_dict['x'] = Independent(Normal(loc=embedding,
                                                         scale=self.scale),
                                                         reinterpreted_batch_ndims=1)
    
    def get_sample(self):
        result_dict = {"x": None}
        result_dict['x'] = self.distribution_dict['x'].rsample()
        return result_dict
    
    def forward(self, embedding):
        self._update_distribution(embedding=embedding)
        return self.distribution_dict


class q_x_class(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int,
                 sigma_dict: dict={"x": 0.1},
                 n_batches: int=1,
                 n_conditions: int=1,
                 n_subjects: int=1) -> None:
        super().__init__()
        
        # x mapping 
        self.loc_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=y_dim,
                              out_features=512,
                              bias=True)),
            ('fc1_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc2', nn.Linear(in_features=512,
                              out_features=256,
                              bias=True)),
            ('fc2_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc3', nn.Linear(in_features=256,
                              out_features=128,
                              bias=True)),
            ('fc3_relu', nn.LeakyReLU(negative_slope=0.2)),
            ('fc4', nn.Linear(in_features=128,
                              out_features=x_dim,
                              bias=True))
        ]))
        
        self.scale = torch.tensor(sigma_dict['x'], dtype=torch.float32)
        
        
        # self.parameter_dict = {"alpha": (nn.Parameter(torch.zeros(1, x_dim, x_dim),
        #                                               requires_grad=False),
        #                                  nn.Parameter(torch.randn(n_batches-1, x_dim, x_dim),
        #                                               requires_grad=True)),
        #                        "beta": (nn.Parameter(torch.zeros(1, x_dim, x_dim),
        #                                               requires_grad=False),
        #                                  nn.Parameter(torch.randn(n_conditions-1, x_dim, x_dim),
        #                                               requires_grad=True)),
        #                        "gamma": (nn.Parameter(torch.zeros(1, x_dim, x_dim),
        #                                               requires_grad=False),
        #                                  nn.Parameter(torch.randn(n_subjects-1, x_dim, x_dim),
        #                                               requires_grad=True))}
        # self.parameter_dict = nn.ParameterDict({"z_alpha_0": nn.Parameter(torch.zeros(1, x_dim, y_dim),
        #                                                                 requires_grad=False),
        #                                         "z_alpha": nn.Parameter(torch.randn(n_batches-1, x_dim, y_dim),
        #                                                               requires_grad=True),
        #                                         "z_beta_0": nn.Parameter(torch.zeros(1, x_dim, y_dim),
        #                                                                requires_grad=False),
        #                                         "z_beta": nn.Parameter(torch.randn(n_conditions-1, x_dim, y_dim),
        #                                                              requires_grad=True),
        #                                         "z_gamma_0": nn.Parameter(torch.zeros(1, x_dim, y_dim),
        #                                                                 requires_grad=False),
        #                                         "z_gamma": nn.Parameter(torch.randn(n_subjects-1, x_dim, y_dim),
        #                                                               requires_grad=True),
        #                                         "w_alpha_0": nn.Parameter(torch.zeros(1, x_dim, y_dim),
        #                                                                 requires_grad=False),
        #                                         "w_alpha": nn.Parameter(torch.randn(n_batches-1, x_dim, y_dim),
        #                                                               requires_grad=True),
        #                                         "w_beta_0": nn.Parameter(torch.zeros(1, x_dim, y_dim),
        #                                                                requires_grad=False),
        #                                         "w_beta": nn.Parameter(torch.randn(n_conditions-1, x_dim, y_dim),
        #                                                              requires_grad=True),
        #                                         "w_gamma_0": nn.Parameter(torch.zeros(1, x_dim, y_dim),
        #                                                                 requires_grad=False),
        #                                         "w_gamma": nn.Parameter(torch.randn(n_subjects-1, x_dim, y_dim),
        #                                                               requires_grad=True)})
        
        self.z_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=y_dim,
                                out_features=x_dim,
                                bias=True))
        ]))
        
        self.w_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=y_dim,
                                out_features=x_dim,
                                bias=True))
        ]))
        
        self.distribution_dict = {"x": None}
        
        
    def _update_distribution(self, 
                             z: torch.tensor, 
                             w: torch.tensor,
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
        
        
        n_batch = torch.sum(FB, dim=0, keepdim=False).unsqueeze(1)
        n_batch[n_batch < 1] = 1
        n_condition = torch.sum(FC, dim=0, keepdim=False).unsqueeze(1)
        n_condition[n_condition < 1] = 1
        n_subject = torch.sum(RS, dim=0, keepdim=False).unsqueeze(1)
        n_subject[n_subject < 1] = 1
        # Batch
        batch_mean_z = torch.einsum("nb, nm -> bm", FB, z)/n_batch
        # batch_mean_z_diff = batch_mean_z - batch_mean_z[0,:]
        batch_mean_free_z = z-torch.einsum("nb, bm -> nm", FB, batch_mean_z)
        
        batch_var_z = torch.einsum("nb, nm -> bm", FB, torch.pow(batch_mean_free_z, 2))/n_batch
        batch_sd_z = torch.sqrt(batch_var_z)
        # batch_var_z_ratio = batch_var_z/(batch_var_z[0,:])
        batch_free_z = batch_mean_free_z/torch.einsum("nb, bm -> nm", FB,  batch_sd_z)
        # Condition
        condition_mean_z = torch.einsum("nc, nm -> cm", FC, batch_free_z)/n_condition
        condition_mean_free_z = batch_free_z-torch.einsum("nc, cm -> nm", FC, condition_mean_z)
        condition_var_z = torch.einsum("nc, nm -> cm", FC, torch.pow(condition_mean_free_z, 2))/n_condition
        condition_sd_z = torch.sqrt(condition_var_z)
        
        condition_free_z = condition_mean_free_z/torch.einsum("nc, cm -> nm", FC,  condition_sd_z)
        # Subject
        subject_mean_z = torch.einsum("ns, nm -> sm", RS, condition_free_z)/n_subject
        subject_mean_free_z = condition_free_z-torch.einsum("ns, sm -> nm", RS, subject_mean_z)
        
        subject_var_z = torch.einsum("ns, nm -> sm", RS, torch.pow(subject_mean_free_z, 2))/n_subject
        subject_sd_z = torch.sqrt(subject_var_z)
        
        effect_free_z = subject_mean_free_z/torch.einsum("ns, sm -> nm", RS,  subject_sd_z)
        
        
        # Batch
        batch_mean_w = torch.einsum("nb, nm -> bm", FB, w)/n_batch
        batch_mean_free_w = z-torch.einsum("nb, bm -> nm", FB, batch_mean_w)
        batch_var_w = torch.einsum("nb, nm -> bm", FB, torch.pow(batch_mean_free_w, 2))/n_batch
        batch_sd_w = torch.sqrt(batch_var_w)
        batch_free_w = batch_mean_free_w/torch.einsum("nb, bm -> nm", FB,  batch_sd_w)
        # Condition
        condition_mean_w = torch.einsum("nc, nm -> cm", FC, batch_free_w)/n_condition
        condition_mean_free_w = batch_free_w-torch.einsum("nc, cm -> nm", FC, condition_mean_w)
        condition_var_w = torch.einsum("nc, nm -> cm", FC, torch.pow(condition_mean_free_w, 2))/n_condition
        condition_sd_w = torch.sqrt(condition_var_w)
        
        condition_free_w = condition_mean_free_w/torch.einsum("nc, cm -> nm", FC,  condition_sd_w)
        # Subject
        subject_mean_w = torch.einsum("ns, nm -> sm", RS, condition_free_w)/n_subject
        subject_mean_free_w = condition_free_w-torch.einsum("ns, sm -> nm", RS, subject_mean_w)
        
        subject_var_w = torch.einsum("ns, nm -> sm", RS, torch.pow(subject_mean_free_w, 2))/n_subject
        subject_sd_w = torch.sqrt(subject_var_w)
        
        effect_free_w = subject_mean_free_w/torch.einsum("ns, sm -> nm", RS,  subject_sd_w)
               
        # z_approx_x = self.z_mapping(z)
        # w_approx_x = self.w_mapping(w)
        # z_extra_loc = torch.einsum("nb, bmd, nd -> nm", FB, alpha_z_mu_sample, z_approx_x) +\
        #     torch.einsum("nc, cmd, nd -> nm", FC, beta_z_mu_sample, z_approx_x) + \
        #     torch.einsum("ns, smd, nd -> nm", RS, gamma_z_mu_sample, z_approx_x)
        # z_extra_sigma = F.softplus(torch.einsum("nb, bmd, nd -> nm", FB, alpha_z_Sigma_sample, z_approx_x) +\
        #     torch.einsum("nc, cmd, nd -> nm", FC, beta_z_Sigma_sample, z_approx_x) + \
        #     torch.einsum("ns, smd, nd -> nm", RS, gamma_z_Sigma_sample, z_approx_x), beta=1)+0.00001
        # w_extra_loc = torch.einsum("nb, bmd, nd -> nm", FB, alpha_w_mu_sample, w_approx_x) +\
        #     torch.einsum("nc, cmd, nd -> nm", FC, beta_w_mu_sample, w_approx_x) + \
        #     torch.einsum("ns, smd, nd -> nm", RS, gamma_w_mu_sample, w_approx_x)
        
        # equal_mean_z = z - z_extra_loc
        # base_mean_z = torch.mean(equal_mean_z, dim=0, keepdim=True)
        # zero_mean_z = equal_mean_z - base_mean_z
        
        # effect_free_w = w - w_extra_loc
        # effect_free_z = zero_mean_z/z_extra_sigma + base_mean_z
        
        w_temp = torch.round(F.sigmoid(effect_free_w))
        wq = effect_free_w + (w_temp - effect_free_w).detach()
        
        effect_free_y = F.softplus(effect_free_z)
        denoised_y = wq * effect_free_y
        
        x = self.loc_mapping(denoised_y)
        # x = x+\
        #     torch.einsum("nb, bmd, nd -> nm", FB, torch.cat((self.parameter_dict['alpha_0'], self.parameter_dict['alpha']), dim=0), x) + \
        #     torch.einsum("nc, cmd, nd -> nm", FC, torch.cat((self.parameter_dict['beta_0'], self.parameter_dict['beta']), dim=0), x) + \
        #     torch.einsum("ns, smd, nd -> nm", RS, torch.cat((self.parameter_dict['gamma_0'], self.parameter_dict['gamma']), dim=0), x)
        
        self.distribution_dict['x'] = Independent(Normal(loc=x,
                                                         scale=self.scale),
                                                         reinterpreted_batch_ndims=1)
        
    def get_sample(self):
        result_dict = {"x": None}
        result_dict['x'] = self.distribution_dict['x'].rsample()
        return result_dict
        
    def forward(self, 
                z: torch.tensor, 
                w: torch.tensor,
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
        self._update_distribution(z=z,
                                  w=w,
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
        
    