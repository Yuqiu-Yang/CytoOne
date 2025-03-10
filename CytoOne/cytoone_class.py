import torch 
import torch.nn as nn 
import torch.nn.functional as F


from CytoOne.generator import encoder, decoder





class cytoone(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim: int=10,
                 encoder_hidden_dims=[500, 500, 2000],
                 decoder_hidden_dims=[2000, 500, 500],
                 n_clusters: int=20):
        super().__init__()

        self.encoder = encoder(input_dim=input_dim,
                               latent_dim=latent_dim,
                               hidden_dims=encoder_hidden_dims)
        self.decoder = decoder(output_dim=input_dim,
                               latent_dim=latent_dim,
                               hidden_dims=decoder_hidden_dims)
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.pi = nn.Parameter(torch.ones(n_clusters) / n_clusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.zeros(n_clusters, latent_dim), requires_grad=True)
        self.log_var_c = nn.Parameter(torch.zeros(n_clusters, latent_dim), requires_grad=True)
        # self.domain_embedding = nn.Embedding(n_batch, 8)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_z, log_var_z = self.encoder(x)
        z = self.reparameterize(mu=mu_z,
                                log_var=log_var_z)

        mu_x, log_var_x = self.decoder(z)
        x_recon = self.reparameterize(mu=mu_x,
                                      log_var=log_var_x)
    
    def compute_gamma(self, z, mu, log_var):
        """
        Compute the posterior probability of z belonging to each cluster
        """
        batch_size = z.size(0)
        
        # Compute log p(z|c) for all c
        z_mu = z.unsqueeze(1) - self.mu_c.unsqueeze(0)  # [batch_size, n_clusters, latent_dim]
        log_var_c = self.log_var_c.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_clusters, latent_dim]
        
        # Compute log p(z|c) = log N(z; mu_c, var_c)
        logpzc = -0.5 * torch.sum(
            log_var_c + torch.pow(z_mu, 2) / torch.exp(log_var_c),
            dim=2
        )  # [batch_size, n_clusters]
        
        # Add log p(c) = log pi_c
        logpc = torch.log(F.softmax(self.pi, dim=0) + 1e-10)
        logpzc += logpc.unsqueeze(0)  # [batch_size, n_clusters]
        
        # Compute gamma = p(c|z) using softmax
        gamma = F.softmax(logpzc, dim=1)  # [batch_size, n_clusters]
        
        return gamma    


