import os 
import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim
from CytoOne.encoder import Encoder
from CytoOne.decoder import Decoder

from CytoOne.utilities import import_data, generate_strata,\
                             load_stratum, reparameterize, \
                             kl_standard, mmd_loss

from torch.distributions import Normal
from CytoOne.basic_distributions import ZeroInflatedSoftplusNormal

from tqdm.auto import tqdm 
from typing import Optional, Union


class cytoone(nn.Module):
    def __init__(self,
                 batch_index_col: Optional[str]=None,
                 celltype_col: Optional[str]=None,
                 normalize: bool=True,
                 dr: bool=True, 
                 zero_inflated: bool=True,
                 latent_dims: list=[10, 2],
                 batch_embedding_dim: int=2, 
                 encoder_hidden_dims: list=[[512, 256], [256, 128]],
                 decoder_hidden_dims: list=[[128, 256], [256, 512]],
                 drop_out_p: float=0.2,
                 model_device: Optional[Union[str, torch.device]] = None):
        super().__init__()

        self.import_data_par = {"batch_index_col": batch_index_col,
                                "celltype_col": celltype_col,
                                "normalize": normalize,
                                "dr": dr,
                                "zero_inflated": zero_inflated}
        self.adata = None
        self.n_batches = None

        self.encoder_par = {"input_dim": None,
                            "batch_embedding_dim": batch_embedding_dim, 
                            "latent_dims": latent_dims,
                            "hidden_dims": encoder_hidden_dims,
                            "drop_out_p": drop_out_p}
        
        self.decoder_par = {"input_dim": None,
                            "batch_embedding_dim": batch_embedding_dim,  
                            "latent_dims": latent_dims[::-1],
                            "hidden_dims": decoder_hidden_dims,
                            "drop_out_p": drop_out_p,
                            "zero_inflated": zero_inflated}
        
        self.zero_inflated = zero_inflated

        # Set model device
        if model_device is None:
            self.model_device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(model_device, str):
            self.model_device = torch.device(model_device)
        else:
            self.model_device = model_device

        self.encoder = None
        self.decoder = None

        self.optimizer = None
        self.batch_embedding = None

        self.log_interval = 10

    def import_data(self,
                    cell_by_gene: Union[str, pd.DataFrame],
                    cell_metadata: Union[str, pd.DataFrame]):
        self.adata = import_data(cell_by_gene=cell_by_gene,
                                 cell_metadata=cell_metadata,
                                 **self.import_data_par)
        
        self.encoder_par['input_dim'] = self.adata.uns["n_genes"]
        self.decoder_par['input_dim'] = self.adata.uns["n_genes"]
        self.n_batches = self.adata.uns['n_batches']

        self.encoder = Encoder(**self.encoder_par)
        self.decoder = Decoder(**self.decoder_par)

        self.batch_embedding = nn.Embedding(self.n_batches, self.encoder_par['batch_embedding_dim'])
        
        self.optimizer = optim.Adam([{'params': self.encoder.parameters()},
                                    {'params': self.decoder.parameters()}], lr=1e-3)

        self.to(self.model_device)

    def encode(self,
               cell_by_gene_counts,
               source_batch_index):
        # Encoder will generate the mu and log_var of the top-level z
        # xs is a list of output of residule blocks
        mu, log_var, xs = self.encoder(x=cell_by_gene_counts,
                                        batch_index=source_batch_index,
                                        batch_embedding=self.batch_embedding)
        # Randomly sample top-level z 
        z = reparameterize(mu, torch.exp(0.5 * log_var))
        return mu, log_var, xs, z

    def decode(self,
               z,
               target_batch_index,
               xs):
        # Based on the zero inflated, we use different likelihood 
        if self.zero_inflated:
            x_mu, x_log_var, x_gate_logit, kl_losses, zs = self.decoder(z=z,
                                                                    batch_index=target_batch_index,
                                                                    batch_embedding=self.batch_embedding,
                                                                    xs=xs,
                                                                    mode='random') 
            x_dists = ZeroInflatedSoftplusNormal(loc=x_mu,
                                                scale=torch.exp(0.5*x_log_var),
                                                gate_logits=x_gate_logit)
        else: 
            x_mu, x_log_var, kl_losses, zs = self.decoder(z=z,
                                                        batch_index=target_batch_index,
                                                        batch_embedding=self.batch_embedding,
                                                        xs=xs,
                                                        mode='random') 
            x_dists = Normal(loc=x_mu,
                            scale=torch.exp(0.5*x_log_var)) 

        return x_dists, kl_losses, zs

    def forward(self,
                cell_by_gene_counts,
                source_batch_index,
                target_batch_index):
        mu, log_var, xs, z = self.encode(cell_by_gene_counts=cell_by_gene_counts,
                                         source_batch_index=source_batch_index) 
        x_dists, kl_losses, zs = self.decode(z=z,
                                             target_batch_index=target_batch_index,
                                             xs=xs)
        # KL divergence top-level 
        kl_losses.append(kl_standard(mu=mu, log_var=log_var)) 
        return x_dists, kl_losses, zs

    def loss_function(self,
                      x_dists: torch.distributions.Distribution,
                      cell_by_gene_counts: torch.tensor,
                      kl_losses: list,
                      zs: list,
                      source_batch_index: torch.tensor):
        
        # Likelihood 
        log_likelihood = x_dists.log_prob(cell_by_gene_counts).sum(dim=1).mean()

        kl_loss = 0
        for k in kl_losses:
            kl_loss += k
        
        # MMD 
        batch_loss = 0
        for z in zs:
            batch_l = mmd_loss(z=z, batch_index=source_batch_index)
            batch_loss += batch_l

        return -log_likelihood + kl_loss + batch_loss


    def training_loop(self,
                      n_epoches: int=20,
                      n_strata: int=100):
        self.train()
        for epoch in range(n_epoches):
            # For epoch, we randomly suffule the data and stratify 
            adata_w_batch_strata = generate_strata(adata=self.adata,
                                                   n_strata=n_strata)
            train_loss = 0.0
            for minibatch_ind in range(n_strata):
                cell_by_gene_counts, source_batch_index, target_batch_index = load_stratum(adata_w_batch_strata=adata_w_batch_strata,
                                                                                           target_batch_index=None,
                                                                                           stratum_id=minibatch_ind,
                                                                                           model_device=self.model_device)
                x_dists, kl_losses, zs = self(cell_by_gene_counts=cell_by_gene_counts,
                                              source_batch_index=source_batch_index,
                                              target_batch_index=target_batch_index)
                
                self.optimizer.zero_grad()
                loss = self.loss_function(x_dists=x_dists,
                                          cell_by_gene_counts=cell_by_gene_counts,
                                          kl_losses=kl_losses,
                                          zs=zs,
                                          source_batch_index=source_batch_index)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                # We gradually decrease the temperature so that the posterior would approach a bernoulli 
                # if minibatch_ind % 100 == 1:
                #     self.current_temperature = np.maximum(self.current_temperature * np.exp(-self.ANNEAL_RATE * minibatch_ind), self.min_temperature) 
                # Print training information 
                if minibatch_ind % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                            epoch, minibatch_ind, n_strata,
                                100. * minibatch_ind / n_strata,
                                loss.item()))
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, train_loss / n_strata))
    

    
    