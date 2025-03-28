import torch 
import torch.nn as nn 
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 batch_embedding_dim: int=8, 
                 latent_dim: int=10,
                 hidden_dims=[500, 500, 2000]):
        super().__init__()
        
        # Build encoder layers
        modules = []
        
        # Input layer
        modules.append(nn.Sequential(
            nn.Linear(input_dim + batch_embedding_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU()
        ))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            modules.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.LeakyReLU()
            ))
        
        self.encoder = nn.Sequential(*modules)
        
        # Mean and log variance projections
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x, 
                batch_index, 
                batch_embedding):
        if batch_embedding is not None:
            batch_emb = batch_embedding(batch_index)
            x = torch.cat([x, batch_emb], dim=1)
        # Get encoder output
        x = self.encoder(x)
        
        # Get mean and log variance
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        return mu, log_var


class decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 batch_embedding_dim: int=8, 
                 latent_dim: int=10,
                 hidden_dims=[2000, 500, 500]):
        super().__init__()
        
        # Build decoder layers
        modules = []
        
        # Input layer
        modules.append(nn.Sequential(
            nn.Linear(latent_dim+batch_embedding_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU()
        ))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            modules.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.LeakyReLU()
            ))
        
        self.decoder = nn.Sequential(*modules)
        
        # Mean and log variance projections
        self.mu = nn.Linear(hidden_dims[-1], output_dim)
        self.log_var = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z, 
                batch_index, 
                batch_embedding):
        if batch_embedding is not None:
            batch_emb = batch_embedding(batch_index)
            z = torch.cat([z, batch_emb], dim=1)
        # Get encoder output
        z = self.decoder(z)
        
        # Get mean and log variance
        mu = self.mu(z)
        log_var = self.log_var(z)

        return mu, log_var
    
class generator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 batch_embedding_dim: int=8, 
                 latent_dim: int=10,
                 encoder_hidden_dims=[500, 500, 2000],
                 decoder_hidden_dims=[2000, 500, 500]) :
        super().__init__()

        self.encoder = encoder(input_dim=input_dim,
                               batch_embedding_dim=batch_embedding_dim,
                               latent_dim=latent_dim,
                               hidden_dims=encoder_hidden_dims)
        self.decoder = decoder(output_dim=output_dim,
                               batch_embedding_dim=batch_embedding_dim,
                               latent_dim=latent_dim,
                               hidden_dims=decoder_hidden_dims)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, source_batch_index, target_batch_index, batch_embedding, compute_source):
        mu_z, log_var_z = self.encoder(x=x, 
                                    batch_index=source_batch_index,
                                    batch_embedding=batch_embedding)
        z = self.reparameterize(mu=mu_z,
                                log_var=log_var_z)
        
        mu_x = None
        log_var_x = None
        if compute_source:
            mu_x, log_var_x = self.decoder(z=z,
                                        batch_index=source_batch_index,
                                        batch_embedding=batch_embedding)
        if (batch_embedding is not None) and (target_batch_index is not None):
            mu_x_target, log_var_x_target = self.decoder(z=z,
                                                        batch_index=target_batch_index,
                                                        batch_embedding=batch_embedding)
            
            x_target = self.reparameterize(mu=mu_x_target,
                                        log_var=log_var_x_target)
        

        return x_target, z, mu_z, log_var_z, mu_x, log_var_x