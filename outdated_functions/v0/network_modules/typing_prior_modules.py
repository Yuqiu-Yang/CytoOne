import torch 
import torch.nn as nn 
import torch.nn.functional as F


class pi_prior_module(nn.Module):
    def __init__(self,
                 n_cell_types: int=10) -> None:
        super().__init__()
        self.logit_p = nn.Parameter(torch.randn(n_cell_types),
                                    requires_grad=True)
        
    def forward(self):
        probs = F.softmax(self.logit_p, dim=0)
        return {
            "probs": probs
        }
        