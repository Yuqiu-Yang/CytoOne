import numpy as np 

import torch 
import torch.nn as nn 

from typing import Optional 

class training_class:
    def __init__(self,
                 cytof_model,
                 optimizer_class,
                 **kwargs) -> None:
        
        self.model = cytof_model
        self.optimizer = optimizer_class(cytof_model.parameters(), **kwargs)

    def train(self,
              n_epoches: int, 
              model_path: str,
              training_data_loader,
              validation_data_loader,
              sigma_y: Optional[np.ndarray]=None):
        for epoch in range(1, n_epoches + 1):
            self.model.train()
            for minibatch, (y, FB, FC, RS) in enumerate(training_data_loader):
                distribution_dict = self.model(y=y,
                                               FB=FB,
                                               FC=FC,
                                               RS=RS)
                training_loss = self.model.compute_loss(distribution_dict=distribution_dict)
                
                self.optimizer.zero_grad()
                training_loss.backward()
                self.optimizer.step()
                
                if (epoch % 1 == 0) and (minibatch % 10 == 0):
                    print("Epoch {}, minibatch {}\n".format(epoch, minibatch))
                    print("Training loss is {}\n".format(training_loss.item()))
                    print("="*25)
            
            self.model.eval()
            with torch.no_grad():
                for (y, FB, FC, RS) in validation_data_loader:
                    pass 