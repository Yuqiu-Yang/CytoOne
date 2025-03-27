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