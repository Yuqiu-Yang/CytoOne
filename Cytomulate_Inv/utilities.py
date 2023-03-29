import torch 
import torch.nn as nn
import torch.distributions as d 
from torch.distributions.kl import register_kl
from pyro.distributions import Delta 
from torch.distributions import Normal, Categorical
from Cytomulate_Inv.basic_distributions import zero_inflated_lognormal


@register_kl(Delta, Delta)
def _kl_delta_delta(p, q):
    return -q.log_density 

@register_kl(Delta, zero_inflated_lognormal)
def _kl_delta_ziln(p, q):
    return -q.log_prob(p.v) 

@register_kl(Delta, Categorical)
def _kl_delta_categorical(p, q):
    return -q.log_prob(p.v) 


