import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Categorical, Independent
from pyro.distributions import Delta 
from Cytomulate_Inv.network_modules.typing_posterior_modules import pi_posterior_module


class typing_prior_class(nn.Module):
    def __init__(self,
                 n_cell_types: int=10) -> None:
        super().__init__()
        
        self.logit_p = nn.Parameter(torch.randn(n_cell_types),
                                    requires_grad=True)
        
    def forward(self):
        p = F.softmax(self.logit_p, dim=0)
        return {
            "pi_prior": Categorical(probs=p)
        }


class typing_posterior_class(nn.Module):
    def __init__(self,
                 x_dim: int=2,
                 n_cell_types: int=10,
                 beta: float=0.25) -> None:
        super().__init__()
        self.pi_module = pi_posterior_module(x_dim=x_dim,
                                             n_cell_types=n_cell_types,
                                             beta=beta)

    def forward(self, x):
        pi_dict, embedding, vq_loss = self.pi_module(x)
        
        
        return {
            "pi_posterior": Independent(Delta(**pi_dict), 
                                        reinterpreted_batch_ndims=1),
            "embedding": embedding,
            "vq_loss": vq_loss
        }
                        
                        