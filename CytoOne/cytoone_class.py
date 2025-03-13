import os 
import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from sklearn.mixture import GaussianMixture
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
        self.pi_c = nn.Parameter(torch.ones(n_clusters) / n_clusters, requires_grad=True)
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

        return x, mu_x, log_var_x, z, mu_z, log_var_z

    def loss_function(self, x, mu_x, log_var_x, z, mu_z, log_var_z):
        log_likelihood = torch.mean(self.diag_gaussian_log_prob(x=x, mu=mu_x, log_var=log_var_x))

        gamma_c = self.compute_gamma_c(z=z)

        kl_z = 0.5 * torch.sum((self.mu_c.unsqueeze(0)-mu_z.unsqueeze(1)).pow(2)/torch.exp(self.log_var_c.unsqueeze(0)) + \
                     torch.exp(log_var_z.unsqueeze(1)-self.log_var_c.unsqueeze(0)) + self.log_var_c.unsqueeze(0), dim=2)
        kl_z = torch.sum(gamma_c * kl_z, dim=1)

        kl_z = torch.mean(kl_z - torch.sum(log_var_z+1, dim=1) * 0.5)

        kl_c = torch.mean(torch.sum(torch.log(gamma_c/self.pi_c.unsqueeze(0)+1e-20) * gamma_c, dim=1))

        return -log_likelihood + kl_z + kl_c

    def compute_gamma_c(self, z):
        # output size batch size * n_clusters 
        gmm_log_probs = self.diag_gaussian_mixture_log_prob(x=z, mu_c=self.mu_c, log_var_c=self.log_var_c)

        temp = torch.exp(torch.log(self.pi_c.unsqueeze(0) + 1e-20) + gmm_log_probs)
        gamma_c = temp/(temp.sum(dim=1).view(-1,1))
        return gamma_c

    def diag_gaussian_mixture_log_prob(self, x, mu_c, log_var_c):
        # x size batch_size * dim
        # mu_c/log_var_c size n_clusters * dim
        # output size batch_size * n_clusters  
        G = []
        for c in range(self.n_clusters):
            G.append(self.diag_gaussian_log_prob(x=x, mu=mu_c[c:c+1, :], log_var=log_var_c[c:c+1,:]).view(-1,1))
        return torch.cat(G, dim=1)


    def diag_gaussian_log_prob(self, x, mu, log_var):
        # x size batch_size * dim
        # mu/log_var size 1 * dim or batch_size * dim 
        # output size batch_size * 1
        individual_log_prob = -0.5 * [np.log(2*np.pi) + log_var + (x-mu).pow(2)/torch.exp(log_var)]
        return torch.sum(individual_log_prob, dim=1, keepdim=True)

    def pretrain_generator(self):
        if  not os.path.exists('./pretrain_model.pk'):

            Loss=nn.MSELoss()
            opti=Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()))

            print('Pretraining......')
            epoch_bar=tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L=0
                for x,y in dataloader:
                    if self.args.cuda:
                        x=x.cuda()

                    z,_=self.encoder(x)
                    x_=self.decoder(z)
                    loss=Loss(x,x_)

                    L+=loss.detach().cpu().numpy()

                    opti.zero_grad()
                    loss.backward()
                    opti.step()

                epoch_bar.write('L2={:.4f}'.format(L/len(dataloader)))

            self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())

            Z = []
            Y = []
            with torch.no_grad():
                for x, y in dataloader:
                    if self.args.cuda:
                        x = x.cuda()

                    z1, z2 = self.encoder(x)
                    assert F.mse_loss(z1, z2) == 0
                    Z.append(z1)
                    Y.append(y)

            Z = torch.cat(Z, 0).detach().cpu().numpy()
            Y = torch.cat(Y, 0).detach().numpy()

            gmm = GaussianMixture(n_components=self.args.nClusters, covariance_type='diag')

            pre = gmm.fit_predict(Z)
            print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

            self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
            self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
            self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())

            torch.save(self.state_dict(), './pretrain_model.pk')

        else:


            self.load_state_dict(torch.load('./pretrain_model.pk')) 
    