from copy import deepcopy
from typing import Dict, List

import einops
import torch
from matplotlib import pyplot as plt

from src.i2c.controller import TimeVaryingStochasticPolicy
from src.i2c.distributions import MultivariateGaussian
from src.i2c.temperature_strategies import TemperatureStrategy
from src.models.wrappers import Model
from src.utils.plotting_utils import plot_trajectory_distribution
from src.utils.torch_tools import CudaAble


class PseudoPosteriorSolver(CudaAble):

    metrics: Dict

    def __init__(
        self,
        dim_x: int,
        dim_u: int,
        horizon: int,
        dynamics: Model,
        cost: Model,
        policy_template: TimeVaryingStochasticPolicy,
        update_temperature_strategy: TemperatureStrategy,
    ):
        self.dim_x, self.dim_u = dim_x, dim_u
        self.horizon = horizon
        self.dynamics = dynamics
        self.cost = cost
        self.policy_template = policy_template
        self.update_temperature_strategy = update_temperature_strategy

    def forward_pass(
        self,
        policy: Model,
        initial_state: MultivariateGaussian,
        alpha: torch.Tensor,
        open_loop: bool = True,
    ) -> (List[MultivariateGaussian], MultivariateGaussian):
        s = initial_state
        sa_cost_list = []
        next_s_list = []
        for t in range(self.horizon):
            a = policy.propagate(s, t=t, open_loop=open_loop)
            sa_policy = a.full_joint(reverse=True)
            sa_cost = self.cost.propagate(sa_policy, alpha=alpha)
            sa_cost_list += [sa_cost]
            # update state_action_cost with dynamics correlations
            next_s = self.dynamics.propagate(sa_cost)
            next_s_list += [next_s]
            s = next_s
        return sa_cost_list, next_s_list, s

    def backward_pass(
        self,
        forward_state_action_distribution: List[MultivariateGaussian],
        predicted_state_distribution: List[MultivariateGaussian],
        terminal_state_distribution: List[MultivariateGaussian],
    ):
        dist = []
        future_state_posterior = terminal_state_distribution
        for current_state_action_prior, pred_s_prior in zip(
            reversed(forward_state_action_distribution),
            reversed(predicted_state_distribution),
        ):
            state_action_posterior = self.smoother(
                current_state_action_prior,
                pred_s_prior,
                future_state_posterior,
            )
            # pass the state posterior to previous timestep
            future_state_posterior = state_action_posterior.marginalize(
                slice(0, self.dim_x)
            )
            dist += [state_action_posterior]
        return list(reversed(dist))

    def smoother(
        self,
        current_prior: MultivariateGaussian,
        predicted_prior: MultivariateGaussian,
        future_posterior: MultivariateGaussian,
    ) -> MultivariateGaussian:
        """linear gaussian smoothing
        See https://vismod.media.mit.edu/tech-reports/TR-531.pdf , Section 3.2
        """
        J = torch.linalg.solve(
            predicted_prior.covariance, predicted_prior.cross_covariance
        ).mT
        try:
            diff_mean = future_posterior.mean - predicted_prior.mean
        except:
            breakpoint()
        mean = current_prior.mean + einops.einsum(
            J, diff_mean, "... xu u, ... u -> ... xu"
        )
        diff_cov = future_posterior.covariance - predicted_prior.covariance
        covariance = current_prior.covariance + J.matmul(diff_cov).matmul(J.mT)
        return MultivariateGaussian(
            mean,
            covariance,
            J.matmul(future_posterior.covariance),
            future_posterior.mean,
            future_posterior.covariance,
        )

    def __call__(
        self,
        n_iteration: int,
        initial_state: MultivariateGaussian,
        policy_prior: Model = None,
        plot_posterior: bool = True,
    ):
        # A1: tempstrat can be modified/tweaked
        # A2: forward to get filter is always on feedforward (also prior)
        self.init_metrics()
        # create as many alphas as batches (0:1 -> trailing 1 dimension)
        alpha = 0.0 * torch.ones_like(initial_state.mean[..., 0:1])
        policy_created = False
        if policy_prior:  # run once with prior to initialize policy
            policy = policy_prior
        else:  # create new (blank) policy
            policy = deepcopy(self.policy_template)
            policy_created = True

        for i in range(n_iteration):
            print(f"alpha {i}: {alpha.flatten().cpu()}")
            sa_filter, s_filter, s_T = self.forward_pass(
                policy,
                initial_state,
                alpha,
                open_loop=True,  # optimize only open-loop
            )
            sa_smoother = self.backward_pass(sa_filter, s_filter, s_T)
            # compute sa_policy trajectory (without cost: alpha=0)
            sa_policy_fb, _, _ = self.forward_pass(
                policy,
                initial_state,
                alpha=0.0,
                open_loop=False,  # to evaluate use feedback
            )
            self.compute_metrics(sa_smoother, sa_policy_fb, alpha)

            if not policy_created:  # switch from prior policy to local policy
                policy = deepcopy(self.policy_template)
                policy_created = True

            policy.model.update_from_distribution(sa_smoother)

            alpha = self.update_temperature_strategy(
                self.cost.predict,
                sa_smoother,  # unused for Polyak
                sa_policy_fb,
                current_alpha=alpha,  # unused for Polyak
            )
            if plot_posterior:
                plot_trajectory_distribution(sa_smoother, f"posterior {i}")
                plot_trajectory_distribution(sa_policy_fb, f"rollout {i}")
                plt.show()
        # if plot_posterior:
        #     # plot_trajectory_distribution(sa_filter, f"filter")
        #     plot_trajectory_distribution(sa_smoother, f"posterior")
        #     plt.show()
        return policy, sa_smoother

    def compute_metrics(self, posterior_distribution, policy_distribution, alpha):
        posterior_cost = self.cost.predict(
            torch.stack(tuple(d.mean for d in posterior_distribution), dim=0)
        ).sum(dim=0)
        policy_cost = self.cost.predict(
            torch.stack(tuple(d.mean for d in policy_distribution), dim=0)
        ).sum(dim=0)
        self.metrics["posterior_cost"] += [posterior_cost.flatten().cpu()]
        self.metrics["policy_cost"] += [policy_cost.flatten().cpu()]
        self.metrics["alpha"] += [alpha.flatten().cpu()]

    def init_metrics(self):
        self.metrics = {
            "posterior_cost": [],
            "policy_cost": [],
            "alpha": [],
        }

    def plot_metrics(self):
        fix, axs = plt.subplots(len(self.metrics))
        for i, (k, v) in enumerate(self.metrics.items()):
            axs[i].plot(v.cpu())
            axs[i].set_ylabel(k)

    def to(self, device):
        self.dynamics.to(device)
        self.cost.to(device)
        self.policy_template.to(device)
        self.update_temperature_strategy.to(device)
        return self
