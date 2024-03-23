import torch 
import torch.nn as nn 
import torch.nn.functional as F


class x_prior_module(nn.Module):
    def __init__(self,
                 scale: float=0.1) -> None:
        super().__init__()
        self.scale = torch.tensor(scale, dtype=torch.float32)
        
    def forward(self, embedding):
        return {
            "loc": embedding,
            "scale": self.scale
        } 