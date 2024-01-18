import torch.nn as nn 


class component_base_class(nn.Module):
    def __init__(self,
                 stage_to_change: str,
                 distribution_info: dict) -> None:
        super().__init__()
        
        assert stage_to_change in ["pretrain", "dimension reduction", "clustering", "abundance effect estimation", "expression effect estimation"], "Illegal stage..."
        
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
            param.requires_grad = (stage == self.stage_to_change)
    
    def _update_distributions(self):
        raise NotImplementedError
     
    def get_samples(self):
        result_dict = {dist: None for dist in self.distribution_dict}
        for dist in self.distribution_dict:
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
    
    