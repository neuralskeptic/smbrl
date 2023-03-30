import torch
from typing import Callable, Sequence
from overrides import override
from dataclasses import dataclass
from src.utils.torch_tools import CudaAble
from functools import partial
from abc import abstractmethod
from src.i2c.distributions import MultivariateGaussian


@dataclass
class Model(CudaAble):
    r"""Abstract callable Model wrapper

    Dataclass objects:

    - Callable approx.inf. method which a takes function that also returns (possibly)
      clamped input x_:
        __call__: Callable(x -> x_mu, x_cov, x_), MultivariateGaussian -> MultivariateGaussian
    - Callable model which takes an input x and has a variable interface.
      Implement the interface by wrapping or overwriting __call__
        model.__call__: x, **_ -> *_
    """
    approximate_inference: Callable
    model: Callable

    def __call__(self, x: torch.Tensor, **kw):
        """calls model on input x \n
        override to use kwargs"""
        return self.model(x)

    def call_and_inputs(self, x: torch.Tensor, **kw):  # override to use kwargs
        """calls model on input x and returns result and (unmodified) input \n
        override to use kwargs & to modify input x"""
        res = self.__call__(x, **kw)
        x_ = x  # override to modify inputs in model (e.g. action constraints)
        if isinstance(res, Sequence):  # prevent nested tuples
            return *res, x_
        else:
            return res, x_

    def propagate(self, dist: MultivariateGaussian, **kw) -> MultivariateGaussian:
        """propagate dist through model using approximate inference \n
        override to use kwargs"""
        fun = partial(self.call_and_inputs, **kw)
        return self.approximate_inference(fun, dist)

    @abstractmethod
    def predict(self, x: torch.Tensor, **kw) -> torch.Tensor:
        """returns single prediction for input x \n
        abstract method: override"""
        raise NotImplementedError

    @abstractmethod
    def predict_dist(self, x: torch.Tensor, **kw) -> MultivariateGaussian:
        """returns prediction distribution (mean and cov) for input x \n
        abstract method: override"""
        raise NotImplementedError

    def to(self, device):
        """returns single prediction for input x \n
        override to add or remove fields"""
        self.model.to(device)
        self.approximate_inference.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def train(self):
        self.model.train()
        return self


@dataclass
class InputModifyingModel(Model):
    @override
    def call_and_inputs(self, x, **kw):  # overload to use kwargs
        *res, x_ = self.__call__(x, **kw)  # modifies x (e.g. action constraints)
        return *res, x_


@dataclass
class DeterministicModel(Model):
    @override
    def predict(self, x: torch.Tensor, **kw) -> torch.Tensor:
        """deterministic model prediction"""
        y, x_ = self.call_and_inputs(x, **kw)
        return y

    @override
    def predict_dist(self, x: torch.Tensor, **kw) -> MultivariateGaussian:
        """prediction distribution with zero (!) covariance"""
        y, x_ = self.call_and_inputs(x, **kw)
        return MultivariateGaussian(y, torch.diag_embed(torch.zeros_like(y)))


@dataclass
class StochasticModel(Model):
    @override
    def predict(self, x: torch.Tensor, *, stoch=False, **kw) -> torch.Tensor:
        mu, cov, x_ = self.call_and_inputs(x, **kw)
        if stoch:
            return torch.distributions.MultivariateNormal(mu, cov).sample()
        else:
            return mu

    @override
    def predict_dist(self, x: torch.Tensor, **kw) -> MultivariateGaussian:
        mu, cov, x_ = self.call_and_inputs(x, **kw)
        return MultivariateGaussian(mu, cov)


@dataclass
class StochasticPolicy(StochasticModel):
    pass


@dataclass
class DeterministicPolicy(DeterministicModel):
    pass


@dataclass
class StochasticDynamics(StochasticModel, InputModifyingModel):
    pass


@dataclass
class DeterministicDynamics(DeterministicModel, InputModifyingModel):
    pass
