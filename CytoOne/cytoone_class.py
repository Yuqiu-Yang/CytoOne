import os 
import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim

from sklearn.mixture import GaussianMixture
from CytoOne.utilities import import_data, generate_strata, load_stratum
from CytoOne.generator import generator
from CytoOne.discriminator import discriminator, latent_discriminator

from tqdm.auto import tqdm 
from typing import Optional, Union


class cytoone(nn.Module):
    def __init__(self,
                 batch_index_col: Optional[str]=None,
                 celltype_col: Optional[str]=None,
                 normalize: bool=True,
                 latent_dim: int=10,
                 batch_embedding_dim: int=8, 
                 encoder_hidden_dims=[500, 500, 2000],
                 decoder_hidden_dims=[2000, 500, 500],
                 discriminator_hidden_dims=[512, 256, 128],
                 latent_discriminator_hidden_dims=[512, 256, 128],
                 n_clusters: int=20,
                 model_device: Optional[Union[str, torch.device]] = None):
        super().__init__()

        self.import_data_par = {"batch_index_col": batch_index_col,
                                "celltype_col": celltype_col,
                                "normalize": normalize}
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
        # Set model device
        if model_device is None:
            self.model_device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(model_device, str):
            self.model_device = torch.device(model_device)
        else:
            self.model_device = model_device

        self.pi_c = None
        self.mu_c = None
        self.log_var_c = None
        
        self.generator = None
        self.discriminator = None
        self.latent_discriminator = None

        self.optimizer_G = None
        self.optimizer_D = None
        self.batch_embedding = None

        self.n_critic = 10

    def import_data(self,
                    cell_by_gene: Union[str, pd.DataFrame],
                    cell_metadata: Union[str, pd.DataFrame]):
        self.adata = import_data(cell_by_gene=cell_by_gene,
                                 cell_metadata=cell_metadata,
                                 **self.import_data_par)
        self.input_dim = self.adata.uns["n_genes"]
        self.n_batches = self.adata.uns['n_batches']

        if self.n_batches == 1:
            self.n_critic = 1

        self.logit_pi_c = nn.Parameter(torch.zeros(self.n_clusters), requires_grad=True)
        self.mu_c = nn.Parameter(torch.zeros(self.n_clusters, self.latent_dim), requires_grad=True)
        self.log_var_c = nn.Parameter(torch.zeros(self.n_clusters, self.latent_dim), requires_grad=True)

        self.generator = generator(input_dim=self.input_dim,
                                   output_dim=self.input_dim,
                                   batch_embedding_dim=self.batch_embedding_dim,
                                   latent_dim=self.latent_dim,
                                   encoder_hidden_dims=self.encoder_hidden_dims,
                                   decoder_hidden_dims=self.decoder_hidden_dims)
        self.optimizer_G = optim.Adam([{'params': self.generator.parameters()},
                                       {'params': self.logit_pi_c.parameters()},
                                       {'params': self.mu_c.parameters()},
                                       {'params': self.log_var_c.parameters()}], lr=1e-3)

        if self.n_batches > 1:
            self.batch_embedding = nn.Embedding(self.n_batches, self.batch_embedding_dim)
            self.discriminator = discriminator(input_dim=self.input_dim,
                                                n_batches=self.n_batches,
                                                hidden_dims=self.discriminator_hidden_dims)
            self.latent_discriminator = latent_discriminator(input_dim=self.latent_dim,
                                                            n_batches=self.n_batches,
                                                            hidden_dims=self.latent_discriminator_hidden_dims)
            self.optimizer_D = optim.Adam([{'params': self.discriminator.parameters()},
                                            {'params': self.latent_discriminator.parameters()}], lr=1e-3)

        self.to(self.model_device)


    def training_loop(self,
                      n_epoches: int=20):
        G_recon = nn.L1Loss(reduction='mean')
        lG_recon = nn.L1Loss(reduction='mean')
        self.train()
        for epoch in range(n_epoches):
            adata_w_batch_strata = generate_strata(adata=self.adata, n_splits=100)
            for minibatch in range(100):
                cell_by_gene_counts, source_batch_index, target_batch_index = load_stratum(adata_w_batch_strata=adata_w_batch_strata,
                                                                                            stratum_id=minibatch,
                                                                                            model_device=self.model_device)
                if self.n_batches > 1:                    
                    cell_by_gene_counts_target, z, _, _, _, _ = self.generator(x=cell_by_gene_counts,
                                                                            source_batch_index=source_batch_index,
                                                                            target_batch_index=target_batch_index,
                                                                            batch_embedding=self.batch_embedding,
                                                                            compute_source=False,
                                                                            compute_target=True)
                    # Train discriminator 
                    self.optimizer_D.zero_grad()

                    real_validity, pred_cls = self.discriminator(cell_by_gene_counts)
                    fake_validity, _ = self.discriminator(cell_by_gene_counts_target.detach())

                    loss_D_adv = -torch.mean(real_validity) + torch.mean(fake_validity)

                    loss_D_cls = F.cross_entropy(pred_cls, source_batch_index, reduction='mean')

                    latent_pred_cls = self.latent_discriminator(z.detach()) 

                    loss_lD_cls = F.cross_entropy(latent_pred_cls, source_batch_index, reduction='mean')
                    
                    loss_D = loss_D_adv + loss_D_cls + loss_lD_cls

                    loss_D.backward()
                    self.optimizer_D.step()

                self.optimizer_G.zero_grad()

                if minibatch % self.n_critic == 0:
                    cell_by_gene_counts_target, z, mu_z, log_var_z, mu_x, log_var_x = self.generator(x=cell_by_gene_counts,
                                                                                                    source_batch_index=source_batch_index,
                                                                                                    target_batch_index=target_batch_index,
                                                                                                    batch_embedding=self.batch_embedding,
                                                                                                    compute_source=True,
                                                                                                    compute_target=True)
                    gamma_c = self.compute_gamma_c(z=z)
                    
                    loss_G_adv = 0
                    loss_G_cls = 0 
                    loss_G_recon = 0 
                    loss_G_l_recon = 0 
                    loss_anchor = 0

                    if self.n_batches > 1:
                        cell_by_gene_counts_recon, z_recon, _, _, _, _ = self.generator(x=cell_by_gene_counts_target,
                                                                            source_batch_index=target_batch_index,
                                                                            target_batch_index=source_batch_index,
                                                                            compute_source=False,
                                                                            compute_target=True)
                        
                        fake_validity, pred_cls = self.discriminator(cell_by_gene_counts_target)
                        loss_G_adv = -torch.mean(fake_validity)
                        loss_G_cls = -F.cross_entropy(pred_cls, target_batch_index, reduction='mean')

                        loss_G_recon = G_recon(cell_by_gene_counts, cell_by_gene_counts_recon)
                        loss_G_l_recon = lG_recon(z, z_recon)
                    
                        gamma_c_recon = self.compute_gamma_c(z=z_recon)
                    
                        loss_anchor = 0.5 * torch.mean(torch.sum(torch.log(gamma_c/gamma_c_recon+1e-20) * gamma_c, dim=1)) + \
                                        0.5 * torch.mean(torch.sum(torch.log(gamma_c_recon/gamma_c+1e-20) * gamma_c_recon, dim=1))
                    
                    log_likelihood = torch.mean(self.diag_gaussian_log_prob(x=cell_by_gene_counts, 
                                                                            mu=mu_x, 
                                                                            log_var=log_var_x))
                    # mu_c is n_clusters * latent_dim -> 1 * n_clusters * latent_dim
                    # mu_z is batch_size * latent_dim -> batch_size * 1 * latent_dim
                    kl_z = 0.5 * torch.sum((self.mu_c.unsqueeze(0)-mu_z.unsqueeze(1)).pow(2)/torch.exp(self.log_var_c.unsqueeze(0)) + \
                                torch.exp(log_var_z.unsqueeze(1)-self.log_var_c.unsqueeze(0)) + self.log_var_c.unsqueeze(0), dim=2)
                    kl_z = torch.sum(gamma_c * kl_z, dim=1)

                    kl_z = torch.mean(kl_z - torch.sum(log_var_z+1, dim=1) * 0.5)

                    kl_c = torch.mean(torch.sum((torch.log(gamma_c+1e-20)-self.logit_pi_c.unsqueeze(0)) * gamma_c, dim=1))
                    loss_vi = -log_likelihood + kl_z + kl_c
                    
                    loss_G = loss_vi + loss_G_adv + loss_G_cls + loss_G_recon + loss_G_l_recon + loss_anchor

                    loss_G.backward()
                    self.optimizer_G.step()
    

    # def gradient_penality(self):
    #     gradient, = torch.autograd.grad(outputs=, inputs=, create_graph=True)
    #     return gradient.square().sum([??])

    def compute_gamma_c(self, z):
        # output size batch size * n_clusters 
        gmm_log_probs = self.diag_gaussian_mixture_log_prob(x=z, mu_c=self.mu_c, log_var_c=self.log_var_c)
        # gmm_log_probs is batch_size * n_clusters
        # logit_pi_c needs to be 1 * n_clusters to broadcast
        temp = torch.exp(self.logit_pi_c.unsqueeze(0) + gmm_log_probs)
        # temp and gamma_c is batch_size * n_clusters
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

            opti=optim.Adam({'params': self.generator.parameters()}, lr=1e-3)

            print('Pretraining......')
            adata_w_batch_strata = generate_strata(adata=self.adata, n_splits=100)

            for minibatch in tqdm(range(100)):
                cell_by_gene_counts, source_batch_index, _ = load_stratum(adata_w_batch_strata=adata_w_batch_strata,
                                                                            stratum_id=minibatch,
                                                                            model_device=self.model_device)
                _, _, mu_z, log_var_z, mu_x, log_var_x = self.generator(x=cell_by_gene_counts,
                                                                        source_batch_index=source_batch_index,
                                                                        target_batch_index=source_batch_index,
                                                                        batch_embedding=self.batch_embedding,
                                                                        compute_source=True,
                                                                        compute_target=False)

                
                log_likelihood = torch.mean(self.diag_gaussian_log_prob(x=cell_by_gene_counts, 
                                                                        mu=mu_x, 
                                                                        log_var=log_var_x))
                kl_z = 0.5 * torch.sum(mu_z.pow(2) + torch.exp(log_var_z), dim=1)
                kl_z = torch.mean(kl_z - torch.sum(log_var_z+1, dim=1) * 0.5)

                loss = -log_likelihood + kl_z

                opti.zero_grad()
                loss.backward()
                opti.step()

            Z = []
            with torch.no_grad():
                for minibatch in tqdm(range(100)):
                    cell_by_gene_counts, source_batch_index, _ = load_stratum(adata_w_batch_strata=adata_w_batch_strata,
                                                                                stratum_id=minibatch,
                                                                                model_device=self.model_device)

                    _, _, mu_z, _, _, _ = self.generator(x=cell_by_gene_counts,
                                                        source_batch_index=source_batch_index,
                                                        target_batch_index=source_batch_index,
                                                        batch_embedding=self.batch_embedding,
                                                        compute_source=True,
                                                        compute_target=False)
                    Z.append(mu_z)

            Z = torch.cat(Z, 0).detach().cpu().numpy()

            gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='diag')

            pre = gmm.fit_predict(Z)

            self.logit_pi_c.data = torch.tensor(np.log(gmm.weights_), dtype=torch.float32, device=self.model_device)
            self.mu_c.data = torch.tensor(gmm.means_, dtype=torch.float32, device=self.model_device)
            self.log_var_c.data = torch.log(torch.tensor(gmm.covariances_, dtype=torch.float32, device=self.model_device))

            torch.save(self.state_dict(), './pretrain_model.pk')
        else:
            self.load_state_dict(torch.load('./pretrain_model.pk')) 
    