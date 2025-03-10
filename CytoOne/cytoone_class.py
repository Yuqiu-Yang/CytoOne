import torch 
import torch.nn as nn 
import torch.nn.functional as F


class VADEEncoder(nn.Module):
    """
    Encoder network for VADE (Variational Deep Embedding)
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[512, 256]):
        super().__init__()
        
        # Build encoder layers
        modules = []
        
        # Input layer
        modules.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
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
        
    def forward(self, x):
        # Get encoder output
        x = self.encoder(x)
        
        # Get mean and log variance
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        return mu, log_var


class VADEDecoder(nn.Module):
    """
    Decoder network for VADE
    """
    def __init__(self, latent_dim, output_dim, hidden_dims=[256, 512]):
        super().__init__()
        
        # Build decoder layers
        modules = []
        
        # Input layer
        modules.append(nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
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

    def forward(self, z):
        # Get encoder output
        z = self.decoder(z)
        
        # Get mean and log variance
        mu = self.mu(z)
        log_var = self.log_var(z)

        return mu, log_var


class Discriminator(nn.Module):
    """
    Discriminator network for StarGAN component
    """
    def __init__(self, input_dim, n_batch, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        # Feature extraction layers
        layers = []
        
        # Input layer
        layers.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2)
        ))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LeakyReLU(0.2)
            ))
        
        self.main = nn.Sequential(*layers)
        
        # Source classification (real/fake)
        self.src = nn.Linear(hidden_dims[-1], 1)
        
        # Domain classification
        self.cls = nn.Linear(hidden_dims[-1], n_batch)
        
    def forward(self, x):
        features = self.main(x)
        src_out = self.src(features)
        cls_out = self.cls(features)
        return src_out, cls_out
    






class cytoone(nn.Module):
    def __init__(self):
        super().__init__()

    
        self.domain_embedding = nn.Embedding(n_batch, 8)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std