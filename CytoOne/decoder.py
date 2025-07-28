# PyTorch
import torch 
import torch.nn as nn 
import torch.nn.functional as F
# Utility
from CytoOne.utilities import ResidualBlock, reparameterize, kl_delta
# Typing 
from typing import Tuple, Optional


class Decoder(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 batch_embedding_dim: int, 
                 latent_dims: list,
                 hidden_dims: list,
                 drop_out_p: float) -> None:
        """Initialize decoder

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
            Drop out probability
        """
        super().__init__()

        self.decoder_tower = nn.ModuleList()
        self.condition_z = nn.ModuleList()
        self.condition_xz = nn.ModuleList()
        current_d = latent_dims[0]*2
        for latent_d, hidden_d in zip(latent_dims[1:], hidden_dims[:-1]):
            self.decoder_tower.append(ResidualBlock(in_dim=current_d,
                                                    out_dim=latent_d,
                                                    hidden_dims=hidden_d,
                                                    drop_out_p=drop_out_p))
            # p(z_l | z_(l-1))
            self.condition_z.append(nn.Sequential(
                ResidualBlock(in_dim=latent_d, 
                              out_dim=latent_d,
                              hidden_dims=[latent_d,latent_d],
                              drop_out_p=drop_out_p),
                nn.GELU(),
                nn.Linear(latent_d, 2*latent_d)
            ))
            # p(z_l | x, z_(l-1))
            self.condition_xz.append(nn.Sequential(
                ResidualBlock(in_dim=latent_d*2,
                              out_dim=latent_d,
                              hidden_dims=[latent_d,latent_d],
                              drop_out_p=drop_out_p),
                nn.GELU(),
                nn.Linear(latent_d, 2*latent_d)
            ))
            current_d = latent_d*2
        
        
        self.recon = nn.Sequential(
            ResidualBlock(in_dim=current_d+batch_embedding_dim,
                            out_dim=current_d+batch_embedding_dim,
                            hidden_dims=hidden_dims[-1],
                            drop_out_p=drop_out_p),
            nn.Linear(current_d+batch_embedding_dim, 3*input_dim)
        )
        # Only used for pretrain model 
        self.decoder_elevator = nn.Sequential(
            nn.Linear(latent_dims[0]+batch_embedding_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 3*input_dim)
        )

    def forward(self, 
                z: torch.tensor, 
                batch_index: torch.tensor, 
                batch_embedding: torch.tensor,
                xs: Optional[list]=None, 
                mode: str="random",
                pretrain: bool=False) -> Tuple[torch.tensor, torch.tensor, torch.tensor, list, list]:
        """Forward for decoder 

        Parameters
        ----------
        z : torch.tensor
            Top level embedding 
        batch_index : torch.tensor
            Batch index 
        batch_embedding : torch.tensor
            All batch embeddings 
        xs : Optional[list], optional
            All letent embeddings, by default None
        mode : str, optional
            Whether to use random sampling, by default "random"
        pretrain : bool, optional
            Whether to use pretrain model, by default False

        Returns
        -------
        Tuple[torch.tensor, torch.tensor, torch.tensor, list, list]
            Mu, log-var, zero-inflation gate for reconstructed x. KL losses and all latent embeddings
        """
        b, w = z.shape
        batch_emb = batch_embedding(batch_index)
        zs = [z]
        if pretrain:
            direct_x_mu, direct_x_log_var, direct_x_gate_logit = self.decoder_elevator(torch.cat([z, batch_emb], dim=1)).chunk(3, dim=1)
            return direct_x_mu, direct_x_log_var, direct_x_gate_logit, [], zs
        else:
            decoder_out = torch.zeros(b, w, device=z.device, dtype=z.dtype)
            kl_losses = []
            for i in range(len(self.decoder_tower)):
                z_sample = torch.cat([decoder_out, z], dim=1)
                decoder_out = self.decoder_tower[i](z_sample)

                mu, log_var = self.condition_z[i](decoder_out).chunk(2, dim=1)

                if xs is not None:
                    delta_mu, delta_log_var = self.condition_xz[i](torch.cat([xs[i], decoder_out], dim=1)).chunk(2, dim=1)
                    kl_losses.append(kl_delta(delta_mu, delta_log_var, mu, log_var))
                    mu = mu + delta_mu
                    log_var = log_var + delta_log_var

                if mode == "fix":
                    z = reparameterize(mu, 0)
                else:
                    z = reparameterize(mu, torch.exp(0.5 * log_var))
                zs.append(z)

            decoder_out = torch.cat([decoder_out, z, batch_emb], dim=1)
            
            x_mu, x_log_var, x_gate_logit = self.recon(decoder_out).chunk(3, dim=1)
            return x_mu, x_log_var, x_gate_logit, kl_losses, zs

        

        