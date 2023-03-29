import torch 
import torch.nn as nn 
import torch.nn.functional as F


class pi_posterior_module(nn.Module):
    def __init__(self,
                 x_dim: int=2,
                 n_cell_types: int=10,
                 beta: float=0.25) -> None:
        super().__init__()
        
        self.beta = beta
        self.cell_embeddings = nn.Embedding(num_embeddings=n_cell_types,
                                            embedding_dim=x_dim)
    
    def forward(self, x):
        # x**2 should be an N * x_dim tensor
        # After torch.sum with keepdim=True,
        # the resulting tensor should be N * 1
        
        # The weight of cell_embeddings should be a 
        # P * x_dim tensor. After torch.sum with keepdim=False 
        # the resulting tensor should be P
        
        # Summing these two tensors up would by 
        # broadcasting give us an N * P tensor 
        
        distance = torch.sum(x**2, dim=1, keepdim=True) + \
                        torch.sum(self.cell_embeddings.weight**2, dim=1, keepdim=False) - \
                        2 * torch.matmul(x, self.cell_embeddings.weight.t())
        
        # Now for each row of X, we will find the index 
        # of the closest embedding 
        index = torch.argmin(distance, dim=1).unsqueeze(dim=1)
        
        # Then we convert this index to its one-hot encoding 
        one_hot_encoding = torch.zeros(distance.size(0), 
                                       distance.size(1))
        one_hot_encoding.scatter_(1, index, 1)
        
        quantized_latents = torch.matmul(one_hot_encoding, self.cell_embeddings.weight)
        
        # Compute VQ loss 
        commitment_loss = F.mse_loss(quantized_latents.detach(), x)
        embedding_loss = F.mse_loss(quantized_latents, x.detach())
        
        vq_loss = commitment_loss * self.beta  + embedding_loss
        
        quantized_latents = x + (quantized_latents-x).detach()
        
        return {
            "v": index,
            "log_density": 1
        }, quantized_latents, vq_loss