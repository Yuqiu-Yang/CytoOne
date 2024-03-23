import torch.nn as nn 
from typing import Union

class component_base_class(nn.Module):
    def __init__(self,
                 stage_to_change: Union[str, list, tuple],
                 distribution_info: dict) -> None:
        super().__init__()
        
        if isinstance(stage_to_change, str):
            stage_to_change = set([stage_to_change])
        else:
            stage_to_change = set(stage_to_change)
        
        assert stage_to_change <= set(["pretrain", "dimension reduction", "clustering", "abundance effect estimation", "expression effect estimation"]), "Illegal stage..."
        
        self.stage_to_change = stage_to_change
        
        has_extra_info = False

        self.distribution_dict = {}
        for dist in distribution_info:
            if distribution_info[dist] is not None:
                has_extra_info = True
            self.distribution_dict[dist] = None 
        
        self.distribution_info_dict = {}
        if has_extra_info:
            for dist in distribution_info:
                self.distribution_info_dict[dist] = distribution_info[dist]
        
    def _update_stage(self,
                      stage: str):
        for param in self.parameters():
            param.requires_grad = (stage in self.stage_to_change)
    
    def _update_distributions(self):
        raise NotImplementedError
     
    def get_samples(self,
                    get_mean: bool=False):
        result_dict = {dist: None for dist in self.distribution_dict}
        for dist in self.distribution_dict:
            if get_mean:
                result_dict[dist] = self.distribution_dict[dist].mean 
            else:
                if "rsample" in dir(self.distribution_dict[dist]):
                    result_dict[dist] = self.distribution_dict[dist].rsample()
                elif "sample" in dir(self.distribution_dict[dist]):
                    result_dict[dist] = self.distribution_dict[dist].sample()
                else:
                    raise KeyError("sample or rsample is not a method of {dist}".format(dist=dist))
        return result_dict   
    
    def forward(self,
                **kwargs):
        self._update_distributions(**kwargs)
        return self.distribution_dict


class component_series_class(component_base_class):
    def __init__(self, 
                 var_name: str,
                 series_length: int,
                 sample_from: str,
                 stage_to_change: Union[str, list , tuple]) -> None:
        super().__init__(stage_to_change=stage_to_change,
                         distribution_info={var_name+str(n): None for n in range(series_length)})
        
        assert sample_from in ["0", "T"], "sample_from has to be either 0 or T"
        self.T = series_length - 1
        self.samples = {}
        for n in range(self.T+1):
            self.samples[var_name+str(n)] = None 

        self.sample_from = sample_from
        
    def get_samples(self,
                    use_current_samples: bool=True,
                    get_mean: bool=False,
                    **kwargs):
        if not use_current_samples:
            self._update_distributions(get_mean=get_mean,
                                       **kwargs) 
        return self.samples
    
        
class model_base_class(nn.Module):
    def __init__(self):
        super().__init__()

    def _update_stage(self,
                      stage: str):
        self.current_stage = stage
        for child in self.children():
            child._update_stage(stage=self.current_stage)
    
    def _update_distributions(self):
        raise NotImplementedError

    def forward(self,
                **kwargs):
        distribution_dict = self._update_distributions(**kwargs)
        return distribution_dict   
    
    def compute_loss(self):
        raise NotImplementedError
    
    