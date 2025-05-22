import os 
import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim
from CytoOne.encoder import Encoder
from CytoOne.decoder import Decoder

from CytoOne.utilities import import_data, generate_strata, load_stratum

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
                 encoder_hidden_dims=[500, 500, 2000],
                 decoder_hidden_dims=[2000, 500, 500],
                 model_device: Optional[Union[str, torch.device]] = None):
        super().__init__()

        self.import_data_par = {"batch_index_col": batch_index_col,
                                "celltype_col": celltype_col,
                                "normalize": normalize}
        self.adata = None
        self.input_dim = None
        self.n_batches = None
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

        self.generator = None

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

        self.encoder = Encoder()
        self.decoder = Decoder()
        
        
        self.optimizer = optim.Adam([{'params': self.encoder.parameters()},
                                       {'params': self.decoder.parameters()}], lr=1e-3)


        self.to(self.model_device)


    def training_loop(self,
                      n_epoches: int=20):
        pass 

                
    

    
    