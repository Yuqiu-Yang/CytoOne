import os 
import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from sklearn.mixture import GaussianMixture
from CytoOne.utilities import import_data
from CytoOne.generator import encoder, decoder
from CytoOne.discriminator import discriminator, latent_discriminator

from typing import Optional, Union


class cytoone(nn.Module):
    def __init__(self,
                 batch_index_col: Optional[str]=None,
                 celltype_col: Optional[str]=None,
                 normalize: bool=True,
                 cofactor: float=5.0,
                 latent_dim: int=10,
                 batch_embedding_dim: int=8, 
                 encoder_hidden_dims=[500, 500, 2000],
                 decoder_hidden_dims=[2000, 500, 500],
                 discriminator_hidden_dims=[512, 256, 128],
                 latent_discriminator_hidden_dims=[512, 256, 128],
                 n_clusters: int=20):
        super().__init__()

        self.import_data_par = {"batch_index_col": batch_index_col,
                                "celltype_col": celltype_col,
                                "normalize": normalize,
                                "cofactor": cofactor}
        self.adata = None
        self.input_dim = None
        self.n_batches = None
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim

        self.batch_embedding_dim = batch_embedding_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.discriminator_hidden_dims = discriminator_hidden_dims
        self.latent_discriminator_hidden_dims = latent_discriminator_hidden_dims

        self.pi_c = nn.Parameter(torch.ones(n_clusters) / n_clusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.zeros(n_clusters, latent_dim), requires_grad=True)
        self.log_var_c = nn.Parameter(torch.zeros(n_clusters, latent_dim), requires_grad=True)
        
        self.batch_embedding = None

    def import_data(self,
                    cell_by_gene: Union[str, pd.DataFrame],
                    cell_metadata: Union[str, pd.DataFrame]):
        self.adata = import_data(cell_by_gene=cell_by_gene,
                                 cell_metadata=cell_metadata,
                                 **self.import_data_par)
        self.input_dim = self.adata.uns["n_genes"]
        self.n_batches = self.adata.uns['n_batches']

        self.encoder = encoder(input_dim=self.input_dim,
                               latent_dim=self.latent_dim,
                               hidden_dims=self.encoder_hidden_dims)
        self.decoder = decoder(output_dim=self.input_dim,
                               latent_dim=self.latent_dim,
                               hidden_dims=self.decoder_hidden_dims)
        if self.n_batches > 1:
            self.batch_embedding = nn.Embedding(self.n_batches, self.batch_embedding_dim)
        
        self.discriminator = discriminator(input_dim=self.input_dim,
                                           n_batches=self.n_batches,
                                           hidden_dims=self.discriminator_hidden_dims)
        self.latent_discriminator = latent_discriminator(input_dim=self.latent_dim,
                                                         n_batches=self.n_batches,
                                                         hidden_dims=self.latent_discriminator_hidden_dims)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, source_batch_index, target_batch_index):
        mu_z, log_var_z = self.encoder(x=x, 
                                       batch_index=source_batch_index,
                                       batch_embedding=self.batch_embedding)
        z = self.reparameterize(mu=mu_z,
                                log_var=log_var_z)

        mu_x, log_var_x = self.decoder(z=z,
                                       batch_index=source_batch_index,
                                       batch_embedding=self.batch_embedding)
        
        mu_x_target, log_var_x_target = self.decoder(z=z,
                                                     batch_index=target_batch_index,
                                                     batch_embedding=self.batch_embedding)
        x_target = self.reparameterize(mu=mu_x_target,
                                       log_var=log_var_x_target)
        
        mu_z_target, log_var_z_target = self.encoder(x=x_target, 
                                                    batch_index=target_batch_index,
                                                    batch_embedding=self.batch_embedding)
        z_target = self.reparameterize(mu=mu_z_target,
                                        log_var=log_var_z_target)
        
        mu_x_recon, log_var_x_recon = self.decoder(z=z_target,
                                                     batch_index=source_batch_index,
                                                     batch_embedding=self.batch_embedding)
        x_recon = self.reparameterize(mu=mu_x_recon,
                                       log_var=log_var_x_recon)

        return x, mu_x, log_var_x, z, mu_z, log_var_z,\
              x_target, mu_x_target, log_var_x_target, \
              z_target, mu_z_target, log_var_z_target, \
              x_recon, mu_x_recon, log_var_x_recon


    def loss_function(self, x, mu_x, log_var_x, z, mu_z, log_var_z,\
                        x_target, mu_x_target, log_var_x_target, \
                        z_target, mu_z_target, log_var_z_target, \
                        x_recon, mu_x_recon, log_var_x_recon):
        log_likelihood = torch.mean(self.diag_gaussian_log_prob(x=x, mu=mu_x, log_var=log_var_x))

        gamma_c = self.compute_gamma_c(z=z)
        gamma_c_target = self.compute_gamma_c(z=z_target)




        kl_z = 0.5 * torch.sum((self.mu_c.unsqueeze(0)-mu_z.unsqueeze(1)).pow(2)/torch.exp(self.log_var_c.unsqueeze(0)) + \
                     torch.exp(log_var_z.unsqueeze(1)-self.log_var_c.unsqueeze(0)) + self.log_var_c.unsqueeze(0), dim=2)
        kl_z = torch.sum(gamma_c * kl_z, dim=1)

        kl_z = torch.mean(kl_z - torch.sum(log_var_z+1, dim=1) * 0.5)

        kl_c = torch.mean(torch.sum(torch.log(gamma_c/self.pi_c.unsqueeze(0)+1e-20) * gamma_c, dim=1))

        return -log_likelihood + kl_z + kl_c


    def training_loop(self,
                      n_epoches: int):
        self.train()
        for epoch in range(n_epoches):
            
            pass 


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
    