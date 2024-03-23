import torch 
import torch.nn as nn
import torch.nn.functional as F

class batch_effect_module(nn.Module):
    def __init__(self,
                 y_dim: int, 
                 x_dim: int, 
                 n_batches: int,
                 distribution_type: str) -> None:
        super().__init__()
        
        self.b_dict = {"w_prior": {},
                       "z_prior": {},
                       "x_posterior": {},
                       "w_posterior": {},
                       "z_posterior": {},
                       "y": {}}
        
        self.b_dict['x_posterior'] = {"loc": nn.Embedding(num_embeddings=n_batches,
                                                      embedding_dim=x_dim),
                                      "scale": nn.Embedding(num_embeddings=n_batches,
                                                      embedding_dim=x_dim)}
        if distribution_type == "ZILN":
            self.b_dict['z_prior'] = {"loc": nn.Embedding(num_embeddings=n_batches,
                                                      embedding_dim=y_dim),
                                    "scale": nn.Embedding(num_embeddings=n_batches,
                                                        embedding_dim=y_dim),
                                    "gate": nn.Embedding(num_embeddings=n_batches,
                                                      embedding_dim=y_dim)} 
        elif distribution_type == "N":
            pass 
        elif distribution_type == "MU":
            pass 
        else:
            raise NotImplementedError
        self.b_dict['w_prior'] = {"loc": nn.Embedding(num_embeddings=n_batches,
                                                      embedding_dim=y_dim),
                                  "scale": nn.Embedding(num_embeddings=n_batches,
                                                        embedding_dim=y_dim)}
        self.b_dict['w_posterior'] = {"loc": nn.Embedding(num_embeddings=n_batches,
                                                      embedding_dim=y_dim),
                                      "scale": nn.Embedding(num_embeddings=n_batches,
                                                      embedding_dim=y_dim)}
        self.b_dict['z_prior'] = {"loc": nn.Embedding(num_embeddings=n_batches,
                                                      embedding_dim=y_dim),
                                "scale": nn.Embedding(num_embeddings=n_batches,
                                                      embedding_dim=y_dim)}
        self.b_dict['z_posterior'] = {"loc": nn.Embedding(num_embeddings=n_batches,
                                                          embedding_dim=y_dim),
                                      "scale": nn.Embedding(num_embeddings=n_batches,
                                                            embedding_dim=y_dim)}
        self.b_dict['y'] = {}
        
        
    def forward(self, b, module_name):
        batch_dict = self.b_dict[module_name]
        
        batch_one_hot = torch.zeros(b.size(0), batch_dict['loc'].weight.size(0))
        batch_one_hot.scatter_(1, b, 1)
        batch_one_hot[0,:] = 0
        
        effect = torch.matmul(batch_one_hot, batch_dict['loc'].weight)
        
        return effect 
    