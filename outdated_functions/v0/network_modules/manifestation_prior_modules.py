import torch 
import torch.nn as nn 
import torch.nn.functional as F

from collections import OrderedDict

class w_prior_module(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int=2,
                 distribution_type: str="ZILN"
                 ) -> None:
        super().__init__()
        self.distribution_type = distribution_type
        
        if distribution_type == "ZILN":
            # When the distribution is Zero Inflated 
            # w is Delta at 1
            # The Delta distribution in pyro only 
            # has two parameters 
            self.v_mapping = nn.Linear(in_features=x_dim,
                                       out_features=y_dim)
            self.log_density_mapping = nn.Linear(in_features=x_dim,
                                       out_features=y_dim)
            # Zeros out all the weights 
            self.v_mapping.weight.data.zero_()
            self.log_density_mapping.weight.data.zero_()
            # Set bias to 1
            self.v_mapping.bias.data.fill_(1.0)
            self.log_density_mapping.bias.data.fill_(1.0)
            # Set requires_grad to False 
            for param in self.v_mapping.parameters():
                param.requires_grad = False 
            for param in self.log_density_mapping.parameters():
                param.requires_grad = False 
            
        elif distribution_type in ["N", "MU"]:
            self.loc_mapping = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(in_features=x_dim,
                                  out_features=128,
                                  bias=True)),
                ('fc1_relu', nn.ReLU()),
                ('fc2', nn.Linear(in_features=128,
                                  out_features=256,
                                  bias=True)),
                ('fc2_relu', nn.ReLU()),
                ('fc3', nn.Linear(in_features=256,
                                  out_features=512,
                                  bias=True)),
                ('fc3_relu', nn.ReLU()),
                ('fc4', nn.Linear(in_features=512,
                                  out_features=y_dim,
                                  bias=True))
            ]))
            
            self.scale_mapping = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(in_features=x_dim,
                                  out_features=128,
                                  bias=True)),
                ('fc1_relu', nn.ReLU()),
                ('fc2', nn.Linear(in_features=128,
                                  out_features=256,
                                  bias=True)),
                ('fc2_relu', nn.ReLU()),
                ('fc3', nn.Linear(in_features=256,
                                  out_features=512,
                                  bias=True)),
                ('fc3_relu', nn.ReLU()),
                ('fc4', nn.Linear(in_features=512,
                                  out_features=y_dim,
                                  bias=True))
            ]))
        else:
            raise NotImplementedError
        
    def forward(self, x):
        if self.distribution_type == "ZILN":
            return {
                'v': self.v_mapping(x),
                'log_density': self.log_density_mapping(x)
            }
        if self.distribution_type in ["N", "MU"]:
            return {
                'loc': self.loc_mapping(x),
                'scale': F.softplus(self.scale_mapping(x), beta=1) + 0.00001
            }


class z_prior_module(nn.Module):
    def __init__(self,
                 y_dim: int,
                 x_dim: int=2,
                 distribution_type: str="ZILN") -> None:
        super().__init__()
        
        self.distribution_type = distribution_type
        self.loc_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=x_dim,
                                out_features=128,
                                bias=True)),
            ('fc1_relu', nn.ReLU()),
            ('fc2', nn.Linear(in_features=128,
                                out_features=256,
                                bias=True)),
            ('fc2_relu', nn.ReLU()),
            ('fc3', nn.Linear(in_features=256,
                                out_features=512,
                                bias=True)),
            ('fc3_relu', nn.ReLU()),
            ('fc4', nn.Linear(in_features=512,
                                out_features=y_dim,
                                bias=True))
        ]))
        
        self.scale_mapping = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=x_dim,
                                out_features=128,
                                bias=True)),
            ('fc1_relu', nn.ReLU()),
            ('fc2', nn.Linear(in_features=128,
                                out_features=256,
                                bias=True)),
            ('fc2_relu', nn.ReLU()),
            ('fc3', nn.Linear(in_features=256,
                                out_features=512,
                                bias=True)),
            ('fc3_relu', nn.ReLU()),
            ('fc4', nn.Linear(in_features=512,
                                out_features=y_dim,
                                bias=True))
        ]))
        
        if distribution_type == "ZILN":
            # When the distribution is Zero Inflated 
            # z is ZILN
            self.gate_mapping = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(in_features=x_dim,
                                    out_features=128,
                                    bias=True)),
                ('fc1_relu', nn.ReLU()),
                ('fc2', nn.Linear(in_features=128,
                                    out_features=256,
                                    bias=True)),
                ('fc2_relu', nn.ReLU()),
                ('fc3', nn.Linear(in_features=256,
                                    out_features=512,
                                    bias=True)),
                ('fc3_relu', nn.ReLU()),
                ('fc4', nn.Linear(in_features=512,
                                    out_features=y_dim,
                                    bias=True))
        ]))
            
        elif distribution_type not in ["N", "MU"]:
            raise NotImplementedError
        
    def forward(self, x):
        result_dict = {
            'loc': self.loc_mapping(x),
            'scale': F.softplus(self.scale_mapping(x), beta=1) + 0.00001
        }
        if self.distribution_type == "ZILN":
            result_dict['gate'] = F.sigmoid(self.gate_mapping(x))
            
        return result_dict
