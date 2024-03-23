# Numeric manipulation 
import math
import numpy as np 

# PyTorch
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributions as d 
from torch.distributions import constraints
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
    _standard_normal,
)
from torch.nn.functional import softplus

# Pyro 
from pyro.distributions import TorchDistribution
from pyro.distributions.util import broadcast_shape
from pyro.distributions import Gamma as pyro_Gamma
from pyro.distributions import LogNormal as pyro_LogNormal


class mollified_uniform(TorchDistribution):
    arg_constraints = {'uniform_low': constraints.real,
                       'uniform_up': constraints.real,
                       'normal_scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, uniform_low, uniform_up, normal_scale, *, validate_args=None):
        self.uniform_low, self.uniform_up, self.normal_scale = broadcast_all(uniform_low, uniform_up, normal_scale)
        super().__init__(self.uniform_low.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(mollified_uniform, _instance)
        batch_shape = torch.Size(batch_shape)
        new.uniform_low = self.uniform_low.expand(batch_shape)
        new.uniform_up = self.uniform_up.expand(batch_shape)
        new.normal_scale = self.normal_scale.expand(batch_shape)
        super(mollified_uniform, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new 
    
    def log_prob(self, value, offset=0.01):
        if self._validate_args:
            self._validate_sample(value) 
        denum = self.uniform_up - self.uniform_low
        upper_cdf = 0.5 * (1 + torch.erf((self.uniform_up - value) * self.normal_scale.reciprocal() / math.sqrt(2)))
        lower_cdf = 0.5 * (1 + torch.erf((self.uniform_low - value) * self.normal_scale.reciprocal() / math.sqrt(2)))
        return ((upper_cdf - lower_cdf) * denum.reciprocal() + offset).log()
        #return torch.asinh((upper_cdf - lower_cdf) * denum.reciprocal()) - 1

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        # Sample from standard uniform first 
        rand = torch.rand(shape, dtype=self.uniform_low.dtype, device=self.uniform_low.device)
        u = rand * (self.uniform_up - self.uniform_low) + self.uniform_low
        eps = _standard_normal(shape, dtype=self.normal_scale.dtype, device=self.normal_scale.device)
        return u + eps * self.normal_scale
    
    @property
    def mean(self):
        return (self.uniform_up + self.uniform_low)/2

    @property
    def variance(self):
        temp = self.uniform_up + self.uniform_low
        return temp.pow(2)/12 + self.normal_scale.pow(2)


class zero_inflated_positive_distribution(TorchDistribution):
    """
    Generic Zero Inflated positive distribution.

    This can be used directly or can be used as a base class as e.g. for
    :class:`ZeroInflatedPoisson` and :class:`ZeroInflatedNegativeBinomial`.

    :param TorchDistribution base_dist: the base distribution.
    :param torch.Tensor gate: probability of extra zeros given via a Bernoulli distribution.
    :param torch.Tensor gate_logits: logits of extra zeros given via a Bernoulli distribution.
    """

    arg_constraints = {
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }

    def __init__(self, base_dist, *, gate=None, gate_logits=None, validate_args=None):
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
                "ZeroInflatedDistribution expected empty "
                "base_dist.event_shape but got {}".format(base_dist.event_shape)
            )

        self.base_dist = base_dist.expand(batch_shape)
        event_shape = torch.Size()

        super().__init__(batch_shape, event_shape, validate_args)

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    @lazy_property
    def gate(self):
        return logits_to_probs(self.gate_logits)

    @lazy_property
    def gate_logits(self):
        return probs_to_logits(self.gate)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        if "gate" in self.__dict__:
            gate, value = broadcast_all(self.gate, value)
            log_prob = self.base_dist.log_prob(value+0.00001)
            log_prob = torch.where(value == 0, (gate).log(), (-gate).log1p() + log_prob)
        else:
            gate_logits, value = broadcast_all(self.gate_logits, value)
            log_prob_minus_log_gate = -gate_logits + self.base_dist.log_prob(value)
            log_gate = -softplus(-gate_logits)
            log_prob = log_prob_minus_log_gate + log_gate
            zero_log_prob = softplus(log_prob_minus_log_gate) + log_gate
            log_prob = torch.where(value == 0, zero_log_prob, log_prob)
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mask = torch.bernoulli(self.gate.expand(shape)).bool()
            samples = self.base_dist.expand(shape).sample()
            samples = torch.where(mask, samples.new_zeros(()), samples)
        return samples


    @lazy_property
    def mean(self):
        return (1 - self.gate) * self.base_dist.mean

    @lazy_property
    def variance(self):
        return (1 - self.gate) * (
            self.base_dist.mean**2 + self.base_dist.variance
        ) - (self.mean) ** 2

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        gate = self.gate.expand(batch_shape) if "gate" in self.__dict__ else None
        gate_logits = (
            self.gate_logits.expand(batch_shape)
            if "gate_logits" in self.__dict__
            else None
        )
        base_dist = self.base_dist.expand(batch_shape)
        zero_inflated_positive_distribution.__init__(
            new, base_dist, gate=gate, gate_logits=gate_logits, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


class zero_inflated_gamma(zero_inflated_positive_distribution):
    arg_constraints = {
        #"concentration": constraints.positive,
        #"rate": constraints.positive,
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
    support = constraints.greater_than_eq(0)

    def __init__(self, concentration, rate, *, 
                 gate=None, gate_logits=None, validate_args=None):
        base_dist = pyro_Gamma(concentration=concentration, 
                          rate=rate, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )


class zero_inflated_lognormal(zero_inflated_positive_distribution):
    arg_constraints = {
        #"concentration": constraints.positive,
        #"rate": constraints.positive,
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
    support = constraints.greater_than_eq(0)

    def __init__(self, loc, scale, *, 
                 gate=None, gate_logits=None, validate_args=None):
        base_dist = pyro_LogNormal(loc=loc, 
                          scale=scale, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )

