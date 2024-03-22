# PyTorch 
import torch 
import torch.nn.functional as F
from torch.distributions import Normal, Independent

# Base class 
from CytoOne.base_class import component_base_class


class p_y_class(component_base_class):
    def __init__(self,
                 model_device: torch.device,
                 y_dim: int=1,
                 y_scale: float=0.001) -> None:
        """p(y|z, w)

        This component defines how the observations are generated 
        given the latent expressions and zero-inflations 
        
        Parameters
        ----------
        y_dim : int, optional
            The dimension of y. During pretrain stage, we are concerned with 
            reconstructing y. Therefore, we "flatten" ys and thus the dimension is 1.
            However, during training, the dimension should match with the number of 
            protein markers in the dataset, by default 1
        y_scale : float, optional
            This defines how much noise should be used to corrupt the generated 
            observations. This is chosen to prevent -\infty log-likelihood from 
            being generated due to negative values, by default 0.001
        """
        super().__init__(stage_to_change="pretrain",
                         distribution_info={"y": None})
        self.model_device = model_device
        self.y_dim = y_dim
        self.y_scale = torch.tensor(y_scale, 
                                    dtype=torch.float32,
                                    device=self.model_device)
        self.reinterpreted_batch_ndims = 0
        if self.y_dim > 1:
            self.reinterpreted_batch_ndims = 1
    
    def _update_distributions(self,
                              z: torch.tensor,
                              w: torch.tensor) -> None:
        """Update p(y|z, w)

        Parameters
        ----------
        z : torch.tensor
            The latent expressions 
        w : torch.tensor
            The latent zero-inflations 
        """
        # We first transform w to 0 and 1
        w_temp = torch.round(F.sigmoid(w))
        # Like VQ-VAE, we use straight-through estimator
        wq = w + (w_temp - w).detach()
        
        # This generates a log-normal RV 
        z = torch.exp(z)
        # Multiplying two yields a zero-inflated log-normal
        denoised_y = wq * z 
        
        # The actual observations will then be corrupted by a little 
        # Gaussian noise 
        # NOTE: in the future, we will future consider other types of 
        # noise distributions and explore the possibility of inferring 
        # noise level via SGD. For now, we will fix the normal formulation.
        self.distribution_dict['y'] = Independent(Normal(loc=denoised_y,
                                                         scale=self.y_scale),
                                                  reinterpreted_batch_ndims=self.reinterpreted_batch_ndims)

        