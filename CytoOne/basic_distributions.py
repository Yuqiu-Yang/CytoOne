# PyTorch
import torch
import torch.nn as nn 
from torch import Tensor
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
from torch.distributions.transforms import SoftplusTransform

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



class ZeroInflatedPositiveDistribution(TorchDistribution):
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
        return logits_to_probs(self.gate_logits, is_binary=True)

    @lazy_property
    def gate_logits(self):
        return probs_to_logits(self.gate, is_binary=True)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        if "gate" in self.__dict__:
            gate, value = broadcast_all(self.gate, value)
            log_prob = torch.where(value == 0, (gate).log(), (-gate).log1p() + self.base_dist.log_prob(value))
        else:
            gate_logits, value = broadcast_all(self.gate_logits, value)
            log_prob = torch.where(value == 0, 
                                   gate_logits-softplus(gate_logits), 
                                   -gate_logits + self.base_dist.log_prob(value)-softplus(-gate_logits))
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mask = torch.bernoulli(self.gate.expand(shape)).bool()
            samples = self.base_dist.expand(shape).sample()
            samples = torch.where(mask, samples.new_zeros(()), samples)
        return samples

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
        ZeroInflatedPositiveDistribution.__init__(
            new, base_dist, gate=gate, gate_logits=gate_logits, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


class ZeroInflatedSoftplusNormal(ZeroInflatedPositiveDistribution):
    arg_constraints = {
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
    support = constraints.greater_than_eq(0)

    def __init__(self, loc, scale, *, 
                 gate=None, gate_logits=None, validate_args=None):
        base_dist = SoftplusNormal(loc=loc, 
                          scale=scale, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )

