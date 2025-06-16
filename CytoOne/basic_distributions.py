# Quadrature 
import math
import numpy as np 
from numpy.polynomial.hermite import hermgauss 
# PyTorch
import torch
import torch.nn as nn 
from torch.distributions import constraints
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import softplus
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SoftplusTransform, ExpTransform
# Pyro 
from pyro.distributions import TorchDistribution
from pyro.distributions.util import broadcast_shape


class SoftplusNormal(TransformedDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale, validate_args=validate_args)
        super().__init__(base_dist, SoftplusTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SoftplusNormal, _instance)
        return super().expand(batch_shape, _instance=new)


class LogNormal(TransformedDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale, validate_args=validate_args)
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogNormal, _instance)
        return super().expand(batch_shape, _instance=new)
    

class QuasiZeroInflatedPositiveDistribution(TorchDistribution):

    arg_constraints = {
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
        "normal_scale": constraints.greater_than_eq(0),
    }

    def __init__(self, base_dist, *, gate=None, gate_logits=None, 
                 normal_scale=None, quadrature_degree=20, validate_args=None):
        if (gate is None) == (gate_logits is None):
            raise ValueError(
                "Either `gate` or `gate_logits` must be specified, but not both."
            )
        if gate is not None:
            batch_shape = broadcast_shape(gate.shape, base_dist.batch_shape)
            self.gate = gate.expand(batch_shape)
        else:
            batch_shape = broadcast_shape(gate_logits.shape, base_dist.batch_shape)
            self.gate_logits = gate_logits.expand(batch_shape)
        if base_dist.event_shape:
            raise ValueError(
                "QuasiZeroInflatedDistribution expected empty "
                "base_dist.event_shape but got {}".format(base_dist.event_shape)
            )
        if (normal_scale is None) or (normal_scale <= 0):
            self.normal_scale = torch.zeros(1)
            self.is_quasi = False
        else:
            self.normal_scale = normal_scale
            self.is_quasi = True
        # Gauss-Hermite nodes and weights (numpy arrays)
        gh_x, gh_w = hermgauss(quadrature_degree)
        self.gh_x = torch.tensor(gh_x, dtype=torch.float32)  # shape [n]
        self.gh_w = torch.tensor(gh_w, dtype=torch.float32)  # shape [n]
        self.base_dist = base_dist.expand(batch_shape)
        event_shape = torch.Size()

        super().__init__(batch_shape, event_shape, validate_args)

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    @lazy_property
    def gate(self):
        return logits_to_probs(self.gate_logits, is_binary=True)

    @lazy_property
    def gate_logits(self):
        return probs_to_logits(self.gate, is_binary=True)
    
    def zero_normal_log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = self.normal_scale**2
        log_scale = (
            math.log(self.normal_scale)
            if isinstance(self.normal_scale, float)
            else self.normal_scale.log()
        )
        return (
            -((value) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        ) 

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        if not self.is_quasi:
            if "gate" in self.__dict__:
                gate, value = broadcast_all(self.gate, value)
                log_prob = torch.where(value <= 0, (gate).log(), (-gate).log1p() + self.base_dist.log_prob(value+1e-7))
            else:
                gate_logits, value = broadcast_all(self.gate_logits, value)
                log_prob = torch.where(value <= 0, 
                                    gate_logits-softplus(gate_logits), 
                                    -gate_logits + self.base_dist.log_prob(value+1e-7)-softplus(-gate_logits))
        else:
            if "gate" in self.__dict__:
                gate, normal_scale, value = broadcast_all(self.gate, self.normal_scale, value)
                
                gh_x = self.gh_x.view((-1,) + (1,) * value.ndim)  # [n, ...]
                gh_w = self.gh_w.view((-1,) + (1,) * (value.ndim))  # [n, ...]
                # If bernoulli is 0 (p) the prob is that of a normal
                zero_p = (gate.log() + self.zero_normal_log_prob(value=value)).exp()
                # If bernoulli is 1 (1-p), gauss hermite with extended support 
                # for n*p value, this will give us 
                # d_quadrature * n * p tensor
                temp_x = gh_x * math.sqrt(2) * normal_scale + value
                x = torch.where(temp_x>1e-7, temp_x, 1e-9)
                w = torch.where(temp_x>1e-7, gh_w, 0)
                gh_int = (self.base_dist.log_prob(x).exp()*w).sum(dim=0)/math.sqrt(math.pi)
                one_p = (-gate).log1p() + (gh_int+1e-7).log()
                log_prob = (zero_p.exp() + one_p.exp()).log()
            else:
                gate_logits, normal_scale, value = broadcast_all(self.gate_logits, self.normal_scale, value)
                
                gh_x = self.gh_x.view((-1,) + (1,) * value.ndim)  # [n, ...]
                gh_w = self.gh_w.view((-1,) + (1,) * (value.ndim))  # [n, ...]
                # If bernoulli is 0 (p) the prob is that of a normal
                zero_p = gate_logits-softplus(gate_logits) + self.zero_normal_log_prob(value=value)
                # If bernoulli is 1 (1-p), gauss hermite with extended support 
                # for n*p value, this will give us 
                # d_quadrature * n * p tensor
                temp_x = gh_x * math.sqrt(2) * normal_scale + value
                x = torch.where(temp_x>1e-7, temp_x, 1e-9)
                w = torch.where(temp_x>1e-7, gh_w, 0)
                gh_int = (self.base_dist.log_prob(x).exp()*w).sum(dim=0)/math.sqrt(math.pi)
                one_p = -gate_logits -softplus(-gate_logits) + (gh_int+1e-7).log()
                log_prob = (zero_p.exp() + one_p.exp()).log() 
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mask = torch.bernoulli(self.gate.expand(shape)).bool()
            samples = self.base_dist.expand(shape).sample()
            samples = torch.where(mask, samples.new_zeros(()), samples)
            noise = 0.0
            if self.is_quasi:
                noise = torch.normal(0.0, self.normal_scale.expand(shape))
        return samples + noise

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        gate = self.gate.expand(batch_shape) if "gate" in self.__dict__ else None
        gate_logits = (
            self.gate_logits.expand(batch_shape)
            if "gate_logits" in self.__dict__
            else None
        )
        normal_scale = self.normal_scale.expand(batch_shape)
        base_dist = self.base_dist.expand(batch_shape)
        QuasiZeroInflatedPositiveDistribution.__init__(
            new, base_dist, gate=gate, gate_logits=gate_logits,\
            normal_scale=normal_scale, quadrature_degree=len(self.gh_x), \
            validate_args=False
        )
        new._validate_args = self._validate_args
        return new


class QuasiZeroInflatedSoftplusNormal(QuasiZeroInflatedPositiveDistribution):
    arg_constraints = {
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
        "normal_scale": constraints.greater_than_eq(0),
    }
    support = constraints.real

    def __init__(self, loc, scale, *, 
                 gate=None, gate_logits=None, 
                 normal_scale=None, quadrature_degree=20,
                 validate_args=None):
        base_dist = SoftplusNormal(loc=loc, 
                          scale=scale, validate_args=False)
        base_dist._validate_args = validate_args
        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits,
            normal_scale=normal_scale, quadrature_degree=quadrature_degree,
            validate_args=validate_args
        )


class QuasiZeroInflatedLogNormal(QuasiZeroInflatedPositiveDistribution):
    arg_constraints = {
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
        "normal_scale": constraints.greater_than_eq(0),
    }
    support = constraints.real

    def __init__(self, loc, scale, *, 
                 gate=None, gate_logits=None, 
                 normal_scale=None, quadrature_degree=20,
                 validate_args=None):
        base_dist = LogNormal(loc=loc, 
                          scale=scale, validate_args=False)
        base_dist._validate_args = validate_args
        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits,
            normal_scale=normal_scale, quadrature_degree=quadrature_degree,
            validate_args=validate_args
        )
        


# class ZeroInflatedPositiveDistribution(TorchDistribution):

#     arg_constraints = {
#         "gate": constraints.unit_interval,
#         "gate_logits": constraints.real,
#     }

#     def __init__(self, base_dist, *, gate=None, gate_logits=None, validate_args=None):
#         if (gate is None) == (gate_logits is None):
#             raise ValueError(
#                 "Either `gate` or `gate_logits` must be specified, but not both."
#             )
#         if gate is not None:
#             batch_shape = broadcast_shape(gate.shape, base_dist.batch_shape)
#             self.gate = gate.expand(batch_shape)
#         else:
#             batch_shape = broadcast_shape(gate_logits.shape, base_dist.batch_shape)
#             self.gate_logits = gate_logits.expand(batch_shape)
#         if base_dist.event_shape:
#             raise ValueError(
#                 "ZeroInflatedDistribution expected empty "
#                 "base_dist.event_shape but got {}".format(base_dist.event_shape)
#             )

#         self.base_dist = base_dist.expand(batch_shape)
#         event_shape = torch.Size()

#         super().__init__(batch_shape, event_shape, validate_args)

#     @constraints.dependent_property
#     def support(self):
#         return self.base_dist.support

#     @lazy_property
#     def gate(self):
#         return logits_to_probs(self.gate_logits, is_binary=True)

#     @lazy_property
#     def gate_logits(self):
#         return probs_to_logits(self.gate, is_binary=True)

#     def log_prob(self, value):
#         if self._validate_args:
#             self._validate_sample(value)

#         if "gate" in self.__dict__:
#             gate, value = broadcast_all(self.gate, value)
#             log_prob = torch.where(value == 0, (gate).log(), (-gate).log1p() + self.base_dist.log_prob(value))
#         else:
#             gate_logits, value = broadcast_all(self.gate_logits, value)
#             log_prob = torch.where(value == 0, 
#                                    gate_logits-softplus(gate_logits), 
#                                    -gate_logits + self.base_dist.log_prob(value+1e-7)-softplus(-gate_logits))
#         return log_prob

#     def sample(self, sample_shape=torch.Size()):
#         shape = self._extended_shape(sample_shape)
#         with torch.no_grad():
#             mask = torch.bernoulli(self.gate.expand(shape)).bool()
#             samples = self.base_dist.expand(shape).sample()
#             samples = torch.where(mask, samples.new_zeros(()), samples)
#         return samples

#     def expand(self, batch_shape, _instance=None):
#         new = self._get_checked_instance(type(self), _instance)
#         batch_shape = torch.Size(batch_shape)
#         gate = self.gate.expand(batch_shape) if "gate" in self.__dict__ else None
#         gate_logits = (
#             self.gate_logits.expand(batch_shape)
#             if "gate_logits" in self.__dict__
#             else None
#         )
#         base_dist = self.base_dist.expand(batch_shape)
#         ZeroInflatedPositiveDistribution.__init__(
#             new, base_dist, gate=gate, gate_logits=gate_logits, validate_args=False
#         )
#         new._validate_args = self._validate_args
#         return new


# class ZeroInflatedSoftplusNormal(ZeroInflatedPositiveDistribution):
#     arg_constraints = {
#         "gate": constraints.unit_interval,
#         "gate_logits": constraints.real,
#     }
#     support = constraints.greater_than_eq(0)

#     def __init__(self, loc, scale, *, 
#                  gate=None, gate_logits=None, validate_args=None):
#         base_dist = SoftplusNormal(loc=loc, 
#                           scale=scale, validate_args=False)
#         base_dist._validate_args = validate_args

#         super().__init__(
#             base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
#         )
