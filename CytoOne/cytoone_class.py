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
                             kl_standard, compute_mmd

from torch.distributions import Normal
from CytoOne.basic_distributions import ZeroInflatedSoftplusNormal

from tqdm.auto import tqdm 
from typing import Optional, Union


class cytoone(nn.Module):
    def __init__(self,
                 batch_index_col: Optional[str]=None,
                 celltype_col: Optional[str]=None,
                 normalize: bool=True,
                 latent_dim: list=[10, 2],
                 batch_embedding_dim: int=2, 
                 encoder_hidden_dims=[[512, 256], [256, 128]],
                 decoder_hidden_dims=[[128, 256], [256, 512]],
                 model_device: Optional[Union[str, torch.device]] = None):
        super().__init__()

        self.import_data_par = {"batch_index_col": batch_index_col,
                                "celltype_col": celltype_col,
                                "normalize": normalize}
        self.adata = None
        self.input_dim = None
        self.n_batches = None
        self.zero_inflated = None
        self.latent_dim = latent_dim

        self.batch_embedding_dim = batch_embedding_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
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

    def import_data(self,
                    cell_by_gene: Union[str, pd.DataFrame],
                    cell_metadata: Union[str, pd.DataFrame]):
        self.adata = import_data(cell_by_gene=cell_by_gene,
                                 cell_metadata=cell_metadata,
                                 **self.import_data_par)
        self.input_dim = self.adata.uns["n_genes"]
        self.n_batches = self.adata.uns['n_batches']
        self.zero_inflated = self.adata.uns['zero_inflated']

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.batch_embedding = nn.Embedding(self.n_batches, self.batch_embedding_dim)
        
        self.optimizer = optim.Adam([{'params': self.encoder.parameters()},
                                       {'params': self.decoder.parameters()}], lr=1e-3)


        self.to(self.model_device)

    def loss_function(self,
                      cell_by_gene_counts,
                      source_batch_index,
                      mu, log_var,
                      x_mu, x_log_var, x_gate_logit):
        
        kl_loss = kl_standard(mu=mu, log_var=log_var) 
        


    def training_loop(self,
                      n_epoches: int=20):
        self.train()
        for epoch in range(n_epoches):
            adata_w_batch_strata = generate_strata(adata=self.adata,
                                                   n_splits=100)
            for minibatch_ind in range(100):
                cell_by_gene_counts, source_batch_index, target_batch_index = load_stratum(adata_w_batch_strata=adata_w_batch_strata,
                                                                                           target_batch_index=None,
                                                                                        stratum_id=minibatch_ind)
                mu, log_var, xs = self.encoder(x=cell_by_gene_counts,
                                            batch_index=source_batch_index,
                                            batch_embedding=self.batch_embedding)
                z = reparameterize(mu, torch.exp(0.5 * log_var))

                if self.zero_inflated:
                    x_mu, x_log_var, x_gate_logit, kl_losses = self.decoder(z=z,
                                                                            batch_index=target_batch_index,
                                                                            batch_embedding=self.batch_embedding,
                                                                            xs=xs,
                                                                            mode='random') 
                else: 
                    x_mu, x_log_var, kl_losses = self.decoder(z=z,
                                                                batch_index=target_batch_index,
                                                                batch_embedding=self.batch_embedding,
                                                                xs=xs,
                                                                mode='random') 

                self.optimizer.zero_grad()
                loss = self.loss_function()
                loss.backward()

                self.optimizer.step()
                
    

    
    