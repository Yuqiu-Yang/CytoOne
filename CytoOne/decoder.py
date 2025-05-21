import torch 
import torch.nn as nn 
import torch.nn.functional as F

from CytoOne.utilities import ResidualBlock, reparameterize, kl


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
                ResidualBlock(in_dim=latent_d, ),
                nn.GELU(),
                nn.Linear()
            ))
            # p(z_l | x, z_(l-1))
            self.condition_xz.append(nn.Sequential(
                ResidualBlock(),
                nn.GELU(),
                nn.Linear()
            ))
            current_d = latent_d
        
        if zero_inflated:
            self.recon = nn.Sequential(
                ResidualBlock(z_dim // 32),
                nn.Conv2d(z_dim // 32, 3, kernel_size=1),
            )
        else:
            self.recon = nn.Sequential(
                ResidualBlock(z_dim // 32),
                nn.Conv2d(z_dim // 32, 3, kernel_size=1),
            ) 

        self.zs = []

    def forward(self, z, xs=None, mode="random", freeze_level=-1):
        """

        :param z: shape. = (B, z_dim, map_h, map_w)
        :return:
        """

        b, h, w = z.shape

        # The init h (hidden state), can be replace with learned param, but it didn't work much
        decoder_out = torch.zeros(b, h, w, device=z.device, dtype=z.dtype)

        kl_losses = []
        if freeze_level != -1 and len(self.zs) == 0 :
            self.zs.append(z)

        for i in range(len(self.decoder_tower)):

            z_sample = torch.cat([decoder_out, z], dim=1)
            decoder_out = self.decoder_tower[i](z_sample)

            if i == len(self.decoder_tower) - 1:
                break

            mu, log_var = self.condition_z[i](decoder_out).chunk(2, dim=1)

            if xs is not None:
                delta_mu, delta_log_var = self.condition_xz[i](torch.cat([xs[i], decoder_out], dim=1)) \
                    .chunk(2, dim=1)
                kl_losses.append(kl(delta_mu, delta_log_var, mu, log_var))
                mu = mu + delta_mu
                log_var = log_var + delta_log_var

            if mode == "fix" and i < freeze_level:
                if len(self.zs) < freeze_level + 1:
                    z = reparameterize(mu, 0)
                    self.zs.append(z)
                else:
                    z = self.zs[i + 1]
            elif mode == "fix":
                z = reparameterize(mu, 0 if i == 0 else torch.exp(0.5 * log_var))
            else:
                z = reparameterize(mu, torch.exp(0.5 * log_var))

            map_h *= 2 ** (len(self.decoder_blocks[i].channels) - 1)
            map_w *= 2 ** (len(self.decoder_blocks[i].channels) - 1)

        if self.zero_inflated:
            x_mu, x_log_var, x_gate_logit = self.recon(decoder_out)
            return x_mu, x_log_var, x_gate_logit, kl_losses
        else:
            x_mu, x_log_var = self.recon(decoder_out)
            return x_mu, x_log_var, kl_losses

        