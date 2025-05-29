import torch 
import torch.nn as nn 
import torch.nn.functional as F

from CytoOne.utilities import ResidualBlock, reparameterize, kl_delta


class Decoder(nn.Module):

    def __init__(self, 
                 input_dim: int,
                 batch_embedding_dim: int=2, 
                 latent_dims: list=[2, 10],
                 hidden_dims: list=[[128, 256], [256, 512]],
                 drop_out_p: float=0.2,
                 zero_inflated: bool=True):
        super().__init__()

        self.zero_inflated=zero_inflated
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
                              hidden_dims=[latent_d,latent_d]),
                nn.GELU(),
                nn.Linear(latent_d, 2*latent_d)
            ))
            # p(z_l | x, z_(l-1))
            self.condition_xz.append(nn.Sequential(
                ResidualBlock(in_dim=latent_d*2,
                              out_dim=latent_d,
                              hidden_dims=[latent_d,latent_d]),
                nn.GELU(),
                nn.Linear(latent_d, 2*latent_d)
            ))
            current_d = latent_d*2
        
        if zero_inflated:
            self.recon = nn.Sequential(
                ResidualBlock(in_dim=current_d+batch_embedding_dim,
                              out_dim=current_d+batch_embedding_dim,
                              hidden_dims=hidden_dims[-1]),
                nn.Linear(current_d+batch_embedding_dim, 3*input_dim)
            )
        else:
            self.recon = nn.Sequential(
                ResidualBlock(in_dim=current_d+batch_embedding_dim,
                              out_dim=current_d+batch_embedding_dim,
                              hidden_dims=hidden_dims[-1]),
                nn.Linear(current_d+batch_embedding_dim, 2*input_dim)
            ) 

    def forward(self, z, 
                batch_index, 
                batch_embedding,
                xs=None, 
                mode="random"):

        b, w = z.shape

        decoder_out = torch.zeros(b, w, device=z.device, dtype=z.dtype)
        
        zs = [z]
        
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

        batch_emb = batch_embedding(batch_index)
        decoder_out = torch.cat([decoder_out, z, batch_emb], dim=1)
        if self.zero_inflated:
            x_mu, x_log_var, x_gate_logit = self.recon(decoder_out).chunk(3, dim=1)
            return x_mu, x_log_var, x_gate_logit, kl_losses, zs
        else:
            x_mu, x_log_var = self.recon(decoder_out).chunk(2, dim=1)
            return x_mu, x_log_var, kl_losses, zs

        

        