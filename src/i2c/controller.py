from dataclasses import dataclass

import einops
import torch
from overrides import override

from src.i2c.distributions import MultivariateGaussian
from src.models.wrappers import StochasticPolicy
from src.utils.torch_tools import CudaAble


@dataclass
class TimeVaryingStochasticPolicy(StochasticPolicy):
    @override
    def __call__(self, x, *, t: int, open_loop=False, **kw):
        mean, cov = self.model.__call__(x, t=t, open_loop=open_loop)
        return mean, cov


class TimeVaryingLinearGaussian(CudaAble):
    def __init__(
        self,
        horizon: int,
        dim_x: int,
        dim_u: int,
        action_covariance: torch.Tensor,
    ):
        assert action_covariance.shape == (dim_u, dim_u)
        self.horizon, self.dim_x, self.dim_u = horizon, dim_x, dim_u
        self.k_actual = torch.tile(
            1e-3 * torch.sin(torch.linspace(0, 20 * torch.pi, horizon))[:, None],
            (1, dim_u),
        )
        self.K_actual = torch.zeros((horizon, dim_u, dim_x))
        self.k_opt, self.K_opt = self.k_actual.clone(), self.K_actual.clone()
        self.sigma = torch.tile(action_covariance, (horizon, 1, 1))
        self.chol = torch.tile(
            torch.linalg.cholesky(action_covariance), (horizon, 1, 1)
        )

    def actual(self):
        from copy import deepcopy

        copy = deepcopy(self)  # crashes: only leaf tensors deepcopy-able
        # copy = TimeVaryingLinearGaussian(
        #     self.horizon,
        #     self.dim_x,
        #     self.dim_u,
        #     self.sigma[0, :, :].detach(),
        # )
        copy.k_opt = self.k_actual.detach()
        copy.K_opt = self.K_actual.detach()
        # TODO skip k/K_actual ?
        return copy

    def __call__(self, x: torch.Tensor, *, t: int, open_loop: bool = False):
        """Accepts (multi-)batch input."""
        # x: (b1, b2, ..., bm, x), K: (t, b1, b2, ..., bn, u, x)
        # k: (t, b1, ..., bn, u), sigma: (t, b1, ..., bn, u, u)
        if open_loop:
            K_t = self.K_opt[t, ...]
            k_t = self.k_opt[t, ...]
        else:
            K_t = self.K_actual[t, ...]
            k_t = self.k_actual[t, ...]
        # unsqueeze gains to match input batch size
        x_batch_dims = len(x.shape) - 1  # m: do not count x
        K_batch_dims = len(K_t.shape) - 2  # n: do not count u,x
        sigma = self.sigma[t, ...]
        for _ in range(x_batch_dims - K_batch_dims):  # for every extra batch dim in x
            K_t = K_t.unsqueeze(-2)  # add 1 dimension before u,x (so dims match)
            k_t = k_t.unsqueeze(-1)  # add 1 dimension before u (so dims match)
            sigma = sigma.unsqueeze(-2)  # add 1 dimension before u,u (so dims match)
        return (
            k_t + einops.einsum(K_t, x, "... u x, ... x -> ... u"),
            torch.tile(sigma, x.shape[K_batch_dims:-1] + (self.dim_u, self.dim_u)),
        )

    # def sample(self, t: int, x: torch.Tensor) -> torch.Tensor:
    #     breakpoint()  # broken!
    #     mu = self.k[t, :] + self.K[t, :, :] @ x
    #     eps = torch.randn(self.dim_u)
    #     return mu + self.chol[t, :, :] @ eps

    def update_from_distribution(self, distribution, *args, **kw):
        assert len(distribution) == self.horizon
        if isinstance(distribution[0], MultivariateGaussian):
            # recreate weights with correct batch dimensions
            device = distribution[0].mean.device
            batch_shape = [self.horizon] + list(distribution[0].mean.shape[:-1])
            bu = batch_shape + [self.dim_u]
            bux = batch_shape + [self.dim_u, self.dim_x]
            buu = batch_shape + [self.dim_u, self.dim_u]
            self.k_actual = torch.empty(bu, device=device)
            self.k_opt = torch.empty(bu, device=device)
            self.K_actual = torch.empty(bux, device=device)
            self.sigma = torch.empty(buu, device=device)
            self.chol = torch.empty(buu, device=device)
            # except K_opt: only unsqueeze (because is not updated = would be overwritten)
            K_opt = self.K_opt
            K_batch_dims = len(self.K_opt.shape) - 2  # do not cound u,x
            # for every extra batch dim
            for _ in range(len(batch_shape) - K_batch_dims):
                # add 1 dimension before u,x (so dims match)
                K_opt = K_opt.unsqueeze(-2)
            self.K_opt = K_opt
            for t, dist in enumerate(distribution):
                x = dist.marginalize(slice(0, self.dim_x))
                u = dist.marginalize(slice(self.dim_x, None))
                K = torch.linalg.solve(
                    x.covariance, dist.covariance[..., : self.dim_x, self.dim_x :]
                ).mT
                self.K_actual[t, ...] = K
                self.k_actual[t, ...] = u.mean - einops.einsum(
                    K, x.mean, "... u x, ... x -> ... u"
                )
                self.k_opt[t, ...] = u.mean - einops.einsum(
                    self.K_opt[t, ...], x.mean, "... u x, ... x -> ... u"
                )
                self.sigma[t, ...] = u.covariance
                self.chol[t, ...] = torch.linalg.cholesky(self.sigma[t, ...])
        else:
            raise ValueError("Cannot update from this distribution!")

    def to(self, device):
        self.k_actual = self.k_actual.to(device)
        self.K_actual = self.K_actual.to(device)
        self.k_opt = self.k_opt.to(device)
        self.K_opt = self.K_opt.to(device)
        self.sigma = self.sigma.to(device)
        self.chol = self.chol.to(device)
        return self
