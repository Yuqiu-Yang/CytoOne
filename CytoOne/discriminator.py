import torch 
import torch.nn as nn 
import torch.nn.functional as F

class discriminator(nn.Module):
    def __init__(self, 
                 input_dim, 
                 n_batches, 
                 hidden_dims=[512, 256, 128]):
        super().__init__()
        
        # Feature extraction layers
        layers = []
        
        # Input layer
        layers.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2)
        ))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LeakyReLU(0.2)
            ))
        
        self.main = nn.Sequential(*layers)
        
        # Source classification (real/fake)
        self.src = nn.Linear(hidden_dims[-1], 1)
        
        # Domain classification
        self.cls = nn.Linear(hidden_dims[-1], n_batches)
        
    def forward(self, x):
        features = self.main(x)
        src_out = self.src(features)
        cls_out = self.cls(features)
        return src_out, cls_out



class latent_discriminator(nn.Module):
    def __init__(self, 
                 input_dim, 
                 n_batches, 
                 hidden_dims=[512, 256, 128]):
        super().__init__()
        
        # Feature extraction layers
        layers = []
        
        # Input layer
        layers.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2)
        ))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LeakyReLU(0.2)
            ))
        
        self.main = nn.Sequential(*layers)
        
        # Domain classification
        self.cls = nn.Linear(hidden_dims[-1], n_batches)
        
    def forward(self, z):
        features = self.main(z)
        cls_out = self.cls(features)
        return cls_out