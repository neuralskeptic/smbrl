import math
from dataclasses import dataclass

import einops
import torch


@dataclass
class Distribution:
    """No idea what this should be yet!"""


@dataclass
class MultivariateGaussian(Distribution):
    """Multivariate Gaussian object for Markovian settings, i.e. p(y) = \int p(x,y) dx = \int p(y | x) p(x) dx."""

    mean: torch.Tensor
    covariance: torch.Tensor
    cross_covariance: torch.Tensor = None
    previous_mean: torch.Tensor = None
    previous_covariance: torch.Tensor = None

    @classmethod
    def from_deterministic(cls, mean: torch.Tensor):
        minimal_cov = torch.diag_embed(1e-8 * torch.ones_like(mean))
        return MultivariateGaussian(mean, minimal_cov)

    def marginalize(self, indices):
        """Marginal of indices"""
        return MultivariateGaussian(
            self.mean[..., indices],
            self.covariance[..., indices, indices],
        )

    def condition_on_mean(self, indices):
        """Conditional of indices conditioned on means (!) of other indices"""
        # get list of complement indices
        other_indices = list(range(self.mean.shape[-1]))  # add all
        other_indices[indices] = []  # remove passed indices
        conditional_mean = self.mean[..., indices]  # ...(x - mean) cancels -> marginal!
        this_cov = self.covariance[..., indices, indices]
        # we can only use one complement index list at once, so we slice twice
        other_cov = self.covariance[..., :, other_indices][..., other_indices, :]
        cross_cov = self.covariance[..., indices, other_indices]  # Cov[this, other]
        conditional_cov = this_cov - cross_cov @ torch.linalg.solve(
            other_cov, cross_cov.mT
        )
        return MultivariateGaussian(
            conditional_mean,
            conditional_cov,
        )

    def conditional(self, indices):
        """Conditional function takes value on which to condition indices"""
        # get list of complement indices
        other_indices = list(range(self.mean.shape[-1]))  # add all
        other_indices[indices] = []  # remove passed indices
        this_cov = self.covariance[..., indices, indices]
        # we can only use one complement index list at once, so we slice twice
        other_cov = self.covariance[..., :, other_indices][..., other_indices, :]
        cross_cov = self.covariance[..., indices, other_indices]  # Cov[this, other]
        conditional_cov = this_cov - cross_cov @ torch.linalg.solve(
            other_cov, cross_cov.mT
        )
        cross_inv_other = torch.linalg.solve(other_cov.mT, cross_cov.mT).mT

        def conditional_f(x: torch.Tensor):
            # mu_y_cond_x = y_mean + Cov[this, other] @ other_cov**-1 @ (x - x_mean)
            this_mean = self.mean[..., indices]
            other_mean = self.mean[..., other_indices]
            other_mean = einops.rearrange(other_mean, "... x -> ... 1 x")  # unsqueeze
            this_mean = einops.rearrange(this_mean, "... y -> ... 1 y")  # unsqueeze
            conditional_mean = this_mean + einops.einsum(
                cross_inv_other, (x - other_mean), "b y x, b p x -> b p y"
            )
            conditional_cov_p = einops.repeat(
                conditional_cov, "b y1 y2 -> b p y1 y2", p=x.shape[1]
            )
            return MultivariateGaussian(conditional_mean, conditional_cov_p)

        return conditional_f

    def full_joint(self, reverse=True):
        """Convert Markovian structure into the joint multivariate Gaussian and forget history."""
        if reverse:
            mean = torch.cat((self.previous_mean, self.mean), axis=-1)
            cross_covariance_T = einops.rearrange(
                self.cross_covariance, "... x1 x2 -> ... x2 x1"
            )
            covariance = torch.cat(
                (
                    torch.cat((self.previous_covariance, cross_covariance_T), axis=-1),
                    (torch.cat((self.cross_covariance, self.covariance), axis=-1)),
                ),
                axis=-2,
            )
        else:
            mean = torch.cat((self.mean, self.previous_mean), axis=-1)
            covariance = torch.cat(
                (
                    torch.cat((self.covariance, self.cross_covariance), axis=-1),
                    (
                        torch.cat(
                            (cross_covariance_T, self.previous_covariance), axis=-1
                        )
                    ),
                ),
                axis=-2,
            )
        return MultivariateGaussian(mean, covariance)

    def get_previous_loglikelihood(self):
        dim = self.previous_mean.shape[0]
        previous_precision = torch.linalg.inv(self.previous_covariance)

        def llh(x):
            diff = x - self.previous_mean
            return -0.5 * dim * (
                math.log(2.0 * math.pi) - torch.logdet(previous_precision)
            ) - 0.5 * torch.einsum("bi,ij,bj->b", diff, previous_precision, diff)

        return llh

    def reverse(self):
        return MultivariateGaussian(
            self.previous_mean,
            self.previous_covariance,
            self.cross_covariance.mT,  # transpose
            self.mean,
            self.covariance,
        )

    def sample(self, *args, **kw):
        """Works for batched mean and covariance"""
        return torch.distributions.MultivariateNormal(
            self.mean, self.covariance
        ).sample(*args, **kw)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.covariance = self.covariance.to(device)
        if self.cross_covariance is not None:
            self.cross_covariance = self.cross_covariance.to(device)
            self.previous_mean = self.previous_mean.to(device)
            self.previous_covariance = self.previous_covariance.to(device)
        return self

    def cpu(self):
        return self.to("cpu")
