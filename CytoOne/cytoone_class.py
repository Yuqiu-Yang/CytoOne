# Data IO 
import os 
import json
# Data manipulation
import numpy as np 
import pandas as pd 
from scipy.stats import ks_2samp
# PyTorch
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim
from torch.distributions import Independent
# Modules 
from CytoOne.encoder import Encoder
from CytoOne.decoder import Decoder
from CytoOne.utilities import import_data, generate_strata,\
                             load_stratum, reparameterize, \
                             kl_standard, mmd_loss, JSONEncoder
from CytoOne.basic_distributions import QuasiZeroInflatedSoftplusNormal
# User entertainment
from tqdm.auto import tqdm 
from typing import Optional, Union, Tuple


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
                 gamma: float=1.0,
                #  anneal_percent: float=0.0,
                 model_device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__()
        # Parameters for importing data 
        self.import_data_par = {"batch_index_col": batch_index_col,
                                "celltype_col": celltype_col,
                                "normalize": normalize,
                                "dr": dr,
                                "zero_inflated": zero_inflated}
        # Parameters for initializing the encoder
        self.encoder_par = {"input_dim": None,
                            "batch_embedding_dim": batch_embedding_dim, 
                            "latent_dims": latent_dims,
                            "hidden_dims": encoder_hidden_dims,
                            "drop_out_p": drop_out_p}
        # Parameters for initializing the decoder
        self.decoder_par = {"input_dim": None,
                            "batch_embedding_dim": batch_embedding_dim,  
                            "latent_dims": latent_dims[::-1],
                            "hidden_dims": decoder_hidden_dims,
                            "drop_out_p": drop_out_p}
        # Data
        self.adata = None
        self.n_batches = None
        self.zero_inflated = zero_inflated
        # Set model device
        if model_device is None:
            self.model_device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(model_device, str):
            self.model_device = torch.device(model_device)
        else:
            self.model_device = model_device
        # Main modules 
        self.encoder = None
        self.decoder = None
        self.rough_log_var = None
        self.noise_log_normal_var = None
        self.batch_embedding = None
        # Optimization 
        self.optimizer = None 
        # self.anneal_percent=anneal_percent
        self.beta = [i/np.max(latent_dims) for i in latent_dims]
        # if self.anneal_percent <= 0.0:
        #     self.beta = 1.0
        # else:
        #     self.beta = 0.0
        self.gamma = gamma
        self.log_interval = 10
        # Monitoring loss 
        self.RECON_list = []
        self.KLD_list = []
        self.MMD_list = []

    def import_data(self,
                    cell_by_gene: Union[str, pd.DataFrame],
                    cell_metadata: Union[str, pd.DataFrame]) -> None:
        self.adata = import_data(cell_by_gene=cell_by_gene,
                                 cell_metadata=cell_metadata,
                                 **self.import_data_par)
        
        self.encoder_par['input_dim'] = self.adata.uns["n_genes"]
        self.decoder_par['input_dim'] = self.adata.uns["n_genes"]
        self.n_batches = self.adata.uns['n_batches']
        if not self.zero_inflated:
            neg_x = self.adata.X[self.adata.X<=0].copy().reshape(-1)
            self.rough_log_var = np.log(np.var(np.concatenate((neg_x, -neg_x))))
        
        
    def initialize_parameters(self):
        self.encoder = Encoder(**self.encoder_par)
        self.decoder = Decoder(**self.decoder_par)

        self.batch_embedding = nn.Embedding(self.n_batches, self.encoder_par['batch_embedding_dim'])
        if not self.zero_inflated:
            self.noise_log_normal_var = nn.Parameter(self.rough_log_var*torch.ones(1), requires_grad=True)
            self.optimizer = optim.Adam([{'params': self.encoder.parameters()},
                                        {'params': self.decoder.parameters()},
                                        {'params': self.batch_embedding.parameters()},
                                        {'params': self.noise_log_normal_var}], lr=1e-3)
        else: 
            self.optimizer = optim.Adam([{'params': self.encoder.parameters()},
                                        {'params': self.decoder.parameters()},
                                        {'params': self.batch_embedding.parameters()}], lr=1e-3)
        self.to(self.model_device)

    def encode(self,
               cell_by_gene_counts: torch.tensor,
               source_batch_index: torch.tensor,
               mode: str="random") -> Tuple[torch.tensor, torch.tensor, list, torch.tensor]:
        # Encoder will generate the mu and log_var of the top-level z
        # xs is a list of output of residule blocks
        mu, log_var, xs = self.encoder(x=cell_by_gene_counts,
                                        batch_index=source_batch_index,
                                        batch_embedding=self.batch_embedding)
        if mode=='fix':
            z = reparameterize(mu, 0)
        else:
            # Randomly sample top-level z 
            z = reparameterize(mu, torch.exp(0.5 * log_var))
        return mu, log_var, xs, z

    def decode(self,
               z: torch.tensor,
               target_batch_index: torch.tensor,
               xs: list,
               mode: str='random',
               denoise: bool=False) -> Tuple[torch.distributions.Distribution, list, list]:
        # Based on the zero inflated, we use different likelihood 
        x_mu, x_log_var, x_gate_logit, kl_losses, zs = self.decoder(z=z,
                                                                batch_index=target_batch_index,
                                                                batch_embedding=self.batch_embedding,
                                                                xs=xs,
                                                                mode=mode) 
        if denoise or self.zero_inflated:
            normal_scale = None
        else:
            normal_scale = torch.exp(0.5*self.noise_log_normal_var)
            
        x_dists = Independent(QuasiZeroInflatedSoftplusNormal(loc=x_mu,
                                            scale=torch.exp(0.5*x_log_var),
                                            gate_logits=x_gate_logit,
                                            normal_scale=normal_scale), 0)

        return x_dists, kl_losses, zs

    def forward(self,
                cell_by_gene_counts: torch.tensor,
                source_batch_index: torch.tensor,
                target_batch_index: torch.tensor) -> Tuple[torch.distributions.Distribution, list, list]:
        mu, log_var, xs, z = self.encode(cell_by_gene_counts=cell_by_gene_counts,
                                         source_batch_index=source_batch_index) 
        x_dists, kl_losses, zs = self.decode(z=z,
                                             target_batch_index=target_batch_index,
                                             xs=xs)
        # KL divergence top-level 
        kl_losses = [kl_standard(mu=mu, log_var=log_var)] + kl_losses
        # kl_losses.append(kl_standard(mu=mu, log_var=log_var)) 
        return x_dists, kl_losses, zs

    def infer(self):
        self.eval()
        with torch.no_grad():
            adata_w_batch = self.adata.to_df().copy()
            adata_w_batch['batch_index'] = self.adata.obs['batch_index'].copy() 
            splits = np.array_split(adata_w_batch.index, 100)
            for i, row_ind in enumerate(splits):
                cell_by_gene_counts = adata_w_batch.loc[row_ind, :]
                source_batch_index = adata_w_batch.loc[row_ind, "batch_index"]
                target_batch_index = source_batch_index
                mu, log_var, xs, z = self.encode(cell_by_gene_counts=cell_by_gene_counts,
                                                source_batch_index=source_batch_index) 
                x_dists, kl_losses, zs = self.decode(z=z,
                                                    target_batch_index=target_batch_index,
                                                    xs=xs)
                
                
    def loss_function(self,
                      x_dists: torch.distributions.Distribution,
                      cell_by_gene_counts: torch.tensor,
                      kl_losses: list,
                      zs: list,
                      source_batch_index: torch.tensor):
        
        # Likelihood 
        log_likelihood = x_dists.log_prob(cell_by_gene_counts).sum(dim=1).mean()

        KLD = 0.0
        for i, k in enumerate(kl_losses[::-1]):
            KLD += k * self.beta[i]
        
        # MMD 
        MMD = 0.0
        if self.n_batches > 1:
            for z in zs:
                batch_l = mmd_loss(z=z, batch_index=source_batch_index)
                MMD += batch_l
        else:
            MMD = torch.zeros(1, dtype=torch.float32)

        return -log_likelihood + KLD + self.gamma*MMD,\
                log_likelihood.detach().cpu().numpy().item(),\
                KLD.detach().cpu().numpy().item(),\
                MMD.detach().cpu().numpy().item()


    def training_loop(self,
                      n_epoches: int=50,
                      n_strata: int=100,
                      early_stop_pval: float=0.5):
        # total_anneal_steps = np.round(n_epoches * n_strata * self.anneal_percent)
        self.train()
        for epoch in range(n_epoches):
            # For epoch, we randomly suffule the data and stratify 
            adata_w_batch_strata = generate_strata(adata=self.adata,
                                                   n_strata=n_strata)
            RECON_epoch_list = []
            KLD_epoch_list = []
            MMD_epoch_list = []
            # Starting from the 3rd epoch, we test if convergence has been achieved 
            if epoch >= 2:
                RECON_previous_2 = np.array(self.RECON_list[epoch-2])
                RECON_previous_1 = np.array(self.RECON_list[epoch-1])
                p_val = ks_2samp(RECON_previous_2, RECON_previous_1).pvalue
                if p_val > early_stop_pval:
                    print("="*30)
                    print("No improvement in the reconstruction task detected. Stop early at epoch {}".format(epoch-1))
                    print("="*30)
                    break
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
                loss, RECON, KLD, MMD = self.loss_function(x_dists=x_dists,
                                                    cell_by_gene_counts=cell_by_gene_counts,
                                                    kl_losses=kl_losses,
                                                    zs=zs,
                                                    source_batch_index=source_batch_index)
                loss.backward()
                RECON_epoch_list.append(RECON)
                KLD_epoch_list.append(KLD)
                MMD_epoch_list.append(MMD)
                train_loss += loss.item()
                self.optimizer.step()
                # # KL Annealing 
                # if self.anneal_percent > 0:
                #     self.beta = np.minimum(1.0, (n_strata*epoch+minibatch_ind)/total_anneal_steps)

                # Print training information 
                if minibatch_ind % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                            epoch, minibatch_ind, n_strata,
                                100. * minibatch_ind / n_strata,
                                loss.item()))
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, train_loss / n_strata))
            self.RECON_list.append(np.array(RECON_epoch_list))
            self.KLD_list.append(np.array(KLD_epoch_list))
            self.MMD_list.append(np.array(MMD_epoch_list))
    
    def save_model(self,
                   dir_name: str,
                   model_name: str):
        torch.save({'model_state_dict': self.state_dict()} | \
                {'optimizer_state_dict': self.optimizer.state_dict()}, 
                os.path.join(dir_name, model_name+".pt")) 

        model_meta = {"import_data_par": self.import_data_par,
                      "encoder_par": self.encoder_par,
                      "decoder_par": self.decoder_par,
                      "n_batches": self.n_batches,
                      "zero_inflated": self.zero_inflated,
                      "rough_log_var": self.rough_log_var}
        with open(os.path.join(dir_name, model_name+"_meta.json"), "w") as f:
            json.dump(model_meta, f, cls=JSONEncoder)
        
    def load_model(self,
                   dir_name: str,
                   model_name: str):
        model_meta = json.load(open(os.path.join(dir_name, model_name+"_meta.json")))
        self.import_data_par = model_meta['import_data_par']
        self.encoder_par = model_meta['encoder_par']
        self.decoder_par = model_meta['decoder_par']
        self.n_batches = model_meta['n_batches']
        self.zero_inflated = model_meta['zero_inflated']
        self.rough_log_var = model_meta["rough_log_var"]
        
        self.initialize_parameters()
        checkpoint = torch.load(os.path.join(dir_name, model_name+".pt")) 
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    