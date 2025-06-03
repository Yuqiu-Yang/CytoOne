import torch 
import torch.nn as nn 
import torch.nn.functional as F
from CytoOne.utilities import ResidualBlock


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 batch_embedding_dim: int=2, 
                 latent_dims: list=[10, 2],
                 hidden_dims: list=[[512, 256], [256, 128]],
                 drop_out_p: float=0.2) -> None:
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

    def forward(self, x, 
                batch_index, 
                batch_embedding,
                pretrain):
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
