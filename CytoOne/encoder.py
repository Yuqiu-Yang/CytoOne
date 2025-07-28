# PyTorch
import torch 
import torch.nn as nn 
import torch.nn.functional as F
# Utility
from CytoOne.utilities import ResidualBlock
# Typing 
from typing import Tuple

class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 batch_embedding_dim: int, 
                 latent_dims: list,
                 hidden_dims: list,
                 drop_out_p: float) -> None:
        """Initialize encoder

        Parameters
        ----------
        input_dim : int
            Dimension of the input 
        batch_embedding_dim : int
            Dimension of batch embedding 
        latent_dims : list
            A list of dimensions of latent variables 
        hidden_dims : list
            A nested list where each sublist contains the number of hidden units 
        drop_out_p : float
            Probability of drop out 
        """
        super().__init__()

        # The encoder module takes x and batch embedding 
        # (x + batch) -> latent_dims[0] -> latent_dims[1] -> ...
        self.encoder_tower = nn.ModuleList()
        current_d = input_dim+batch_embedding_dim
        for latent_d, hidden_d in zip(latent_dims, hidden_dims):
            self.encoder_tower.append(ResidualBlock(in_dim=current_d,
                                                    out_dim=latent_d,
                                                    hidden_dims=hidden_d,
                                                    drop_out_p=drop_out_p))
            current_d = latent_d
        
        # Generate mu and log_var for the top-level z 
        self.condition_x = nn.Sequential(
            nn.GELU(),
            nn.Linear(current_d, 2*current_d)
        )
        # This module directly mapps from X to top-level z
        self.encoder_elevator = nn.Sequential(
            nn.Linear(input_dim+batch_embedding_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 2*current_d)
        )

    def forward(self, 
                x: torch.tensor, 
                batch_index: torch.tensor, 
                batch_embedding: torch.tensor, 
                pretrain: bool) -> Tuple[torch.tensor, torch.tensor, list]:
        """Forward for encoder

        Parameters
        ----------
        x : torch.tensor
            Input 
        batch_index : torch.tensor
            Batch index 
        batch_embedding : torch.tensor
            All batch embeddings 
        pretrain : bool
            If use pretrain model 

        Returns
        -------
        Tuple[torch.tensor, torch.tensor, list]
            Top level mu, log_var, and all other embeddings 
        """
        batch_emb = batch_embedding(batch_index)
        x = torch.cat([x, batch_emb], dim=1)
        direct_mu, direct_log_var = self.encoder_elevator(x).chunk(2, dim=1)
        if pretrain:
            return  direct_mu, direct_log_var, []
        else:
            xs = []
            last_x = x
            for e in self.encoder_tower:
                x = e(x)
                last_x = x
                xs.append(x)

            mu, log_var = self.condition_x(last_x).chunk(2, dim=1)
            # xs is now [latent_dims[0], latent_dims[1], ...]
            # we do not need the top-level for the decoder
            # To make indexing a litter easier, we also reverse the order 
            return direct_mu + 0.1*mu, direct_log_var + 0.1*log_var, xs[:-1][::-1] 
