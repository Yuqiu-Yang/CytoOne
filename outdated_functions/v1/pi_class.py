import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Categorical, Independent
from pyro.distributions import Delta 


class p_pi_class(nn.Module):
    def __init__(self,
                 n_cell_types: int) -> None:
        super().__init__()
        
        self.logit_p = nn.Parameter(torch.randn(1, n_cell_types),
                                    requires_grad=True)

        self.distribution_dict = {"pi": None}
    
    def _update_distribution(self, 
                             beta_pi_sample: torch.tensor,
                             gamma_pi_sample: torch.tensor,
                             FC: torch.tensor,
                             RS: torch.tensor):
        
        condition_effect = torch.einsum("nc, ck -> nk", FC, beta_pi_sample)
        subject_effect = torch.einsum("ns, sk -> nk", RS, gamma_pi_sample)
        logits = self.logit_p + condition_effect + subject_effect
        probs = F.softmax(logits, dim=1)
        self.distribution_dict['pi'] = Independent(Categorical(probs=probs),
                                                   reinterpreted_batch_ndims=0)
    def get_sample(self):
        result_dict = {"pi": None}
        result_dict['pi'] = self.distribution_dict['pi'].sample()
        return result_dict
    
    def forward(self, 
                beta_pi_sample: torch.tensor,
                gamma_pi_sample: torch.tensor,
                FC: torch.tensor,
                RS: torch.tensor):
        self._update_distribution(beta_pi_sample=beta_pi_sample,
                                  gamma_pi_sample=gamma_pi_sample,
                                  FC=FC,
                                  RS=RS)
        
        return self.distribution_dict


class q_pi_class(nn.Module):
    def __init__(self,
                 x_dim: int,
                 n_cell_types: int,
                 vq_vae_weight: float) -> None:
        super().__init__()

        self.vq_vae_weight = vq_vae_weight
        self.cell_embeddings = nn.Embedding(num_embeddings=n_cell_types,
                                            embedding_dim=x_dim)
        self.distribution_dict = {"pi": None,
                                  "embedding": None,
                                  "vq_loss": None}
        
    def _update_distribution(self, x):
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
        
        vq_loss = commitment_loss * self.vq_vae_weight  + embedding_loss
        
        quantized_latents = x + (quantized_latents-x).detach() 
        
        self.distribution_dict['pi'] = Independent(Delta(v=index.squeeze(1),
                                                         log_density=1), 
                                                         reinterpreted_batch_ndims=0)
        self.distribution_dict['embedding'] = quantized_latents
        self.distribution_dict['vq_loss'] = vq_loss
    
    def get_sample(self):
        result_dict = {"pi": None}
        result_dict['pi'] = self.distribution_dict['pi'].sample()
        return result_dict    
    
    def forward(self, x):
        self._update_distribution(x=x)
        return self.distribution_dict
                        
                        