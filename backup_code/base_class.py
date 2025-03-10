# PyTorch 
import torch 
import torch.nn as nn 

# Typing 
from typing import Union

class component_base_class(nn.Module):
    def __init__(self,
                 stage_to_change: Union[str, list, tuple],
                 distribution_info: dict) -> None:
        """The basic class for all model components 

        Parameters
        ----------
        stage_to_change : Union[str, list, tuple]
            Training stages at which the weights can change 
            Namely, requires_grad=True
        distribution_info : dict
            A dictionary whose keys are the variable names. 
            Most of the time the corresponding values are simply None. 
            However for inferring different effects, the values are information 
            on shape of the distributions
        """
        super().__init__()
        # We first convert stage_to_change to a set 
        if isinstance(stage_to_change, str):
            stage_to_change = set([stage_to_change])
        else:
            stage_to_change = set(stage_to_change)
        
        assert stage_to_change <= set(["pretrain", "dimension reduction", "clustering", "abundance effect estimation", "expression effect estimation"]), "Illegal stage..."
        
        self.stage_to_change = stage_to_change
        
        has_extra_info = False
        # distribution_dict will store the actual distributions 
        self.distribution_dict = {}
        for dist in distribution_info:
            if distribution_info[dist] is not None:
                has_extra_info = True
            self.distribution_dict[dist] = None 
        # distribution_info_dict stores the information on the shape of the distributions 
        self.distribution_info_dict = {}
        if has_extra_info:
            for dist in distribution_info:
                self.distribution_info_dict[dist] = distribution_info[dist]
        # samples is a dictionary containing temporary samples 
        # generated from the distributions 
        self.samples = {dist: None for dist in self.distribution_dict}
        
    def _update_stage(self,
                      stage: str) -> None:
        """Update requires_grad according to the current stage 

        Parameters
        ----------
        stage : str
            The current stage 
        """
        # If the current stage is in stage_to_change
        # then the parameters will require gradients 
        for param in self.parameters():
            param.requires_grad = (stage in self.stage_to_change)
    
    def _update_distributions(self) -> None:
        """Defines how distributions are updated 

        Raises
        ------
        NotImplementedError
            This should be individually specified for each model component 
        """
        raise NotImplementedError
     
    def get_samples(self,
                    get_mean: bool=False) -> dict:
        """Generate a sample from each distribution

        Parameters
        ----------
        get_mean : bool, optional
            Whether or not to get the mean parameter of the distribution instead of 
            a random sample, by default False

        Returns
        -------
        dict
            A dictionary containing a random sample from each distribution
        """
        # We won't check if .mean or .sample is a method 
        # If not, PyTorch will throw an error 
        for dist in self.distribution_dict:
            if get_mean:
                self.samples[dist] = self.distribution_dict[dist].mean 
            else:
                if "rsample" in dir(self.distribution_dict[dist]):
                    self.samples[dist] = self.distribution_dict[dist].rsample()
                else:
                    self.samples[dist] = self.distribution_dict[dist].sample()
        return self.samples   
    
    def forward(self,
                **kwargs) -> dict:
        """Forward pass of the component 

        Returns
        -------
        dict
            A dictionary containing all the updated distributions 
        """
        self._update_distributions(**kwargs)
        return self.distribution_dict
 
class model_base_class(nn.Module):
    def __init__(self) -> None:
        """The basic class for all model classes
        
        A model class will contain one or several component objects 
        """
        super().__init__()

    def _update_stage(self,
                      stage: str) -> None:
        """Recursively update stages for all components contained within

        Parameters
        ----------
        stage : str
            The current training stage 
        """
        self.current_stage = stage
        for child in self.children():
            child._update_stage(stage=self.current_stage)
    
    def _update_distributions(self) -> dict:
        """Defines how distributions are updated 

        Returns
        -------
        dict
            A dictionary containing dictionaries of the updated distributions 
            and other information needed for computing loss

        Raises
        ------
        NotImplementedError
            This should be individually specified for each model 
        """
        raise NotImplementedError

    def forward(self,
                **kwargs) -> dict:
        """Forward pass of the model 

        Returns
        -------
        dict
            A dictionary containing dictionaries of the updated distributions 
            and other information needed for computing loss 
        """
        distribution_dict = self._update_distributions(**kwargs)
        return distribution_dict   
    
    def compute_loss(self) -> torch.tensor:
        """Defines how loss should be computed 

        Returns
        -------
        torch.tensor
            A tensor of loss

        Raises
        ------
        NotImplementedError
            This should be individually specified for each model
        """
        raise NotImplementedError
    
    