import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from pyro.distributions import Delta 

from CytoOne.basic_distributions import mollified_uniform

from typing import Optional 


class p_y_class(nn.Module):
    def __init__(self,
                 distribution_type: str="ZILN",
                 sigma_dict: dict={"delta": -1,
                                   "y": 0.01},
                 fixed_noise_level: bool=True) -> None:
        super().__init__()
        
        assert distribution_type in ["ZILN", "N", "MU"], "distribution_type has to be one of ZILN, N, or MU"
        assert sigma_dict.keys() >= {"delta", "y"}, "sigma_dict for y should contain delta and y"
        
        self.distribution_type = distribution_type
        
        self.parameter_dict = nn.ParameterDict({distribution_type+"_delta": nn.Parameter(torch.tensor(sigma_dict['delta'], dtype=torch.float32),
                                                                                         requires_grad= not fixed_noise_level),
                                                distribution_type+"_y": nn.Parameter(torch.tensor(sigma_dict['y'], dtype=torch.float32),
                                                                                     requires_grad=False)})
                
        self.distribution_dict = {"y": None}
    
    def _update_sigma_y(self, 
                        sigma_y: Optional[float]=None):
        if sigma_y is not None:
            self.parameter_dict[self.distribution_type+'_y'] = nn.Parameter(torch.tensor(sigma_y, dtype=torch.float32),
                                                                            requires_grad=False)
    
    def _update_distribution(self,
                             z: torch.tensor,
                             w: torch.tensor,
                             sigma_y: Optional[float]=None):
        
        w_temp = torch.round(F.sigmoid(w))
        wq = w + (w_temp - w).detach()
        
        z = torch.exp(z)
        denoised_y = wq * z 
        
        self._update_sigma_y(sigma_y=sigma_y)
        
        if self.distribution_type == "ZILN":
            self.distribution_dict['y'] = Independent(Delta(v=denoised_y,
                                                            log_density=1),
                                                      reinterpreted_batch_ndims=1)
        if self.distribution_type == "N":
            self.distribution_dict['y'] = Independent(Normal(loc=denoised_y,
                                                             scale=F.softplus(self.parameter_dict['N_delta'], beta=1)+0.00001),
                                                      reinterpreted_batch_ndims=1)
        if self.distribution_type == "MU":
            self.distribution_dict['y'] = Independent(mollified_uniform(uniform_low=denoised_y-F.softplus(self.parameter_dict['MU_delta'], beta=1)-0.00001,
                                                                        uniform_up=denoised_y,
                                                                        normal_scale=self.parameter_dict['MU_y']),
                                                      reinterpreted_batch_ndims=1)
    
    def get_sample(self):
        result_dict = {"y": None}
        result_dict['y'] = self.distribution_dict['y'].rsample()
        return result_dict 
    
    def forward(self,
                z: torch.tensor,
                w: torch.tensor,
                sigma_y: Optional[float]=None):
        self._update_distribution(z=z,
                                  w=w,
                                  sigma_y=sigma_y) 
        return self.distribution_dict

        