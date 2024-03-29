import os
import time
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from experiment_launcher import run_experiment
from experiment_launcher.utils import save_args
from mushroom_rl.core.logger.logger import Logger
from overrides import override
from torch.func import jacrev, vmap
from tqdm import trange

from src.datasets.mutable_buffer_datasets import ReplayBuffer
from src.environments.pendulum import Pendulum
from src.feature_fns.nns import MultiLayerPerceptron, ResidualNetwork
from src.i2c.approximate_inference import (
    CubatureQuadrature,
    GaussHermiteQuadrature,
    QuadratureGaussianInference,
    QuadratureImportanceSamplingInnovation,
    QuadratureInference,
)
from src.i2c.controller import TimeVaryingLinearGaussian, TimeVaryingStochasticPolicy
from src.i2c.distributions import MultivariateGaussian
from src.i2c.solver import PseudoPosteriorSolver
from src.i2c.temperature_strategies import Annealing, Constant, PolyakStepSize
from src.models.linear_bayesian_models import (
    NLM_MLP_RFF,
    NeuralLinearModelMLP,
    NeuralLinearModelResNet,
    SpectralNormalizedNeuralGaussianProcess,
)
from src.models.wrappers import (
    DeterministicCostModel,
    DeterministicDynamics,
    DeterministicPolicy,
    Model,
    StochasticDynamics,
    StochasticPolicy,
)
from src.utils.decorator_factory import Decorator
from src.utils.plotting_utils import (
    add_grid,
    plot_mvn,
    plot_trajectory_distribution,
    rollout_plot,
    scaled_xaxis,
)
from src.utils.seeds import fix_random_seed
from src.utils.time_utils import timestamp

# class HitCounter:
#     spd_counter = 0  # static

#     def spd_fix():
#         HitCounter.spd_counter += 1


# def nearest_spd(covariance):
#     return covariance  # do nothing
#     HitCounter.spd_fix()
#     L, V = torch.linalg.eig(covariance)
#     L[L.real <= 1e-10] = 1e-6  # is it safe to do real?
#     return V @ torch.diag(L) @ V.T


def experiment(
    env_type: str = "localPendulum",  # Pendulum
    horizon: int = 200,
    batch_size: int = 200 * 10,  # lower if gpu out of memory
    n_iter: int = 1,  # outer loop
    train_test_ratio: float = 80 / 20,
    ## frequency or period: whichever is lower dominates
    log_frequency: float = 0.1,  # every p-th of n_epochs
    log_period: int = 20,  # every N epochs
    model_save_frequency: float = 0.5,  # every p-th of n_epochs
    model_save_period: int = 500,  # every N epochs
    ##########
    min_epochs_per_train: int = 1,  # train at least N epochs (even if early stop)
    early_stop_thresh: float = -1e3,  # stop training when training & validation loss lower
    ## dynamics ##
    n_dyn_train_episodes: int = 50,
    plot_dyn: bool = True,  # plot pointwise and rollout prediction
    n_trajs_plot_dyn: int = 1,  # how many trajs in dyn plot [0 => all trajs]
    # #  D1) true dynamics
    # dyn_model_type: str = "env",
    # #  D2) mlp model
    # dyn_model_type: str = "mlp",
    # layer_spec_dyn: int = [*(256,) * 3],  # [in, *layer_spec, out]
    # lr_dyn: float = 5e-4,
    # n_epochs_dyn: int = 300,
    # #  D3) resnet model
    # dyn_model_type: str = "resnet",
    # layer_spec_dyn: int = [*(256,) * 5],  # [in, *layer_spec, out]
    # lr_dyn: float = 3e-4,
    # n_epochs_dyn: int = 1000,
    # # D4) linear regression w/ mlp features
    # dyn_model_type: str = "nlm-mlp",
    # layer_spec_dyn: int = [*(128,) * 2],  # [in, *layer_spec, feat], W: [feat, out]
    # n_features_dyn: int = 128,
    # lr_dyn: float = 1e-4,
    # n_epochs_dyn: int = 1000,
    # # D5) linear regression w/ resnet features
    # dyn_model_type: str = "nlm-resnet",
    # layer_spec_dyn: int = [*(128,) * 2],  # [in, *layer_spec, feat], W: [feat, out]
    # n_features_dyn: int = 128,
    # lr_dyn: float = 1e-4,
    # n_epochs_dyn: int = 1000,
    # D6) linear regression w/ spec.norm.-resnet & rf features
    dyn_model_type: str = "sngp",
    layer_spec_dyn: int = [*(128,) * 5],  # [in, *layer_spec, feat], W: [feat, out]
    # layer_spec_dyn: int = [*(256,) * 3],  # [in, *layer_spec, feat], W: [feat, out]
    n_features_dyn: int = 256,  # RFFs require ~512-1024 for accuracy (but greatly increase NN param #)
    lr_dyn: float = 5e-4,
    n_epochs_dyn: int = 100,
    ##############
    ## policy ##
    pol_strategy: str = "moment matching",  # quadrature augmented moment matching
    # pol_strategy: str = "dagger",  # dagger + importance weighting
    n_epochs_between_rollouts: int = 10,  # dagger policy distillation
    plot_policy: bool = True,  # plot pointwise and rollout prediction
    n_trajs_plot_pol: int = 1,  # how many trajs in policy plot [0 => all trajs]
    # #  D1) local time-varying linear gaussian controllers (i2c)
    # policy_type: str = "tvlg",
    # #  D2) mlp model
    # policy_type: str = "mlp",
    # layer_spec_pol: int = [*(256,) * 3],  # [in, *layer_spec, out]
    # lr_pol: float = 5e-4,
    # n_epochs_pol: int = 300,
    # #  D3) resnet model
    # policy_type: str = "resnet",
    # layer_spec_pol: int = [*(128,) * 8],  # [in, *layer_spec, out]
    # lr_pol: float = 5e-4,
    # n_epochs_pol: int = 500,
    # # D4) linear regression w/ mlp features
    # policy_type: str = "nlm-mlp",
    # layer_spec_pol: int = [256, 256],  # [in, *layer_spec, feat], W: [feat, out]
    # n_features_pol: int = 256,
    # lr_pol: float = 5e-4,
    # n_epochs_pol: int = 300,
    # # D5) linear regression w/ resnet features
    # policy_type: str = "nlm-resnet",
    # layer_spec_pol: int = [*(128,) * 8],  # [in, *layer_spec, feat], W: [feat, out]
    # n_features_pol: int = 256,
    # lr_pol: float = 5e-4,
    # n_epochs_pol: int = 500,
    # D6) linear regression w/ spec.norm.-resnet & rf features
    policy_type: str = "sngp",
    # layer_spec_pol: int = [*(64,) * 8],  # [in, *layer_spec, feat], W: [feat, out]
    layer_spec_pol: int = [*(128,) * 2],  # [in, *layer_spec, feat], W: [feat, out]
    # layer_spec_pol: int = [*(256,) * 2],  # [in, *layer_spec, feat], W: [feat, out]
    n_features_pol: int = 256,  # RFFs require ~512-1024 for accuracy (but greatly increase NN param #)
    lr_pol: float = 1e-4,
    n_epochs_pol: int = 10000,
    # # Dx1) bottleneck mlp with rf features
    # policy_type: str = "nlm_mlp_rff",
    # layer_spec_pol: int = [256, 256, 12],
    # n_features_pol: int = 256,  # RFFs require ~512-1024 for accuracy (but greatly increase NN param #)
    # lr_pol: float = 5e-4,
    # n_epochs_pol: int = 300,
    ############
    ## i2c solver ##
    n_iter_solver: int = 100,  # how many i2c solver iterations to do
    n_i2c_vec: int = 10,  # how many local policies in the vectorized i2c batch
    s0dot_i2c_var: float = 1e-6,  # very low initial velocity variance (low energy)
    s0_i2c_var: float = 1e-6,  # how much initial state variance i2c should start with
    delta_cost_thresh: float = 1,  # if cost change lower stop optim
    # plot_posterior: bool = False,  # plot state-action-posterior over time
    plot_posterior: bool = True,  # plot state-action-posterior over time
    # plot_local_policy: bool = False,  # plot time-cum. sa-posterior cost, local policy cost, and alpha per iter
    plot_local_policy: bool = True,  # plot time-cum. sa-posterior cost, local policy cost, and alpha per iter
    ############
    ## general ##
    show_plots: bool = True,  # if False never plt.show(), but creates and saves
    plotting: bool = True,  # if False not plot creation
    plot_data: bool = False,  # visualize data trajectories (sns => very slow!)
    log_console: bool = True,  # also log to console (not just log file); FORCE on if debug
    log_wandb: bool = True,  # off if debug
    wandb_project: str = "smbrl_i2c",
    wandb_entity: str = "showmezeplozz",
    wandb_job_type: str = "train",
    seed: int = 1,
    results_dir: str = "logs/tmp/",
    debug: bool = True,  # prepends logdir with 'debug/', disables wandb, enables console logging
    # debug: bool = False,  # prepends logdir with 'debug/', disables wandb, enables console logging
    use_cuda: bool = True,  # for policy/dynamics training (i2c always on cpu)
):
    ####################################################################################################################
    #### ~~~~ SETUP (saved to yaml) ~~~~

    if debug:
        # disable wandb logging and redirect normal logging to ./debug directory
        print("@@@@@@@@@@@@@@@@@ DEBUG: LOGGING DISABLED @@@@@@@@@@@@@@@@@")
        os.environ["WANDB_MODE"] = "disabled"  # close terminal to reset
        log_wandb = False
        log_console = True
        results_dir = "debug" / Path(results_dir)

    # Fix seed
    fix_random_seed(seed)

    # Results directory
    wandb_group: str = f"i2c_{env_type}_{dyn_model_type}_{policy_type}"
    repo_dir = Path.cwd().parent
    results_dir = repo_dir / results_dir / wandb_group / str(seed) / timestamp()
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    # plotting config
    fig_kwargs = dict(
        figsize=(12, 9),
        dpi=150,
    )
    if dyn_model_type != "env":
        n_epochs_between_eval_dyn = min(n_epochs_dyn * log_frequency, log_period)
    n_epochs_between_eval_pol = min(n_epochs_pol * log_frequency, log_period)

    # Save arguments
    save_args(results_dir, locals(), git_repo_path="./")

    # logger
    logger = Logger(
        config=locals(),
        log_name=seed,
        results_dir=results_dir,
        project=wandb_project,
        entity=wandb_entity,
        wandb_kwargs={"group": wandb_group, "job_type": wandb_job_type},
        tags=["dyn_model_type", "policy_type", "env_type"],
        log_console=log_console,
        log_wandb=log_wandb,
    )

    ####################################################################################################################
    #### ~~~~ EXPERIMENT SETUP ~~~~

    logger.info(f"Env: {env_type}, Dyn: {dyn_model_type}, Pol: {policy_type}")
    logger.info(f"Seed: {seed}")

    time_begin = time.time()

    # visialization options
    torch.set_printoptions(precision=7)
    torch.set_default_dtype(torch.float64)

    #### ~ S: mdp, initial state, cost
    if env_type == "localPendulum":
        environment = Pendulum()  # local seed
    dim_xu = environment.dim_xu
    dim_x = environment.dim_x
    dim_u = environment.dim_u
    u_max = environment.u_mx
    # initial state variance is part of env, thus should not be hyperparam
    initial_state_distribution = MultivariateGaussian(
        torch.Tensor([torch.pi, 0.0]),
        # low initial velocity variance!!!
        # 1e-6 * torch.eye(dim_x),  # original
        torch.diag_embed(torch.Tensor([1e-2, 1e-6])),  # more exploration
    )

    dyn_train_buffer = ReplayBuffer(
        [dim_xu, dim_x], batchsize=batch_size, device=device, max_size=50 * horizon
    )
    dyn_test_buffer = ReplayBuffer(
        [dim_xu, dim_x], batchsize=batch_size, device=device, max_size=50 * horizon
    )

    pol_train_buffer = ReplayBuffer(
        [dim_x, dim_u, (dim_u, dim_u), 1],  # [s_mean, a_mean, a_cov, importance]
        batchsize=batch_size,
        device=device,
        max_size=50 * horizon,
    )
    pol_test_buffer = ReplayBuffer(
        [dim_x, dim_u, (dim_u, dim_u), 1],  # [s_mean, a_mean, a_cov, importance]
        batchsize=batch_size,
        device=device,
        max_size=50 * horizon,
    )

    #### ~ S: approximate inference params
    quad_params = CubatureQuadrature(1, 0, 0)
    gh_params = GaussHermiteQuadrature(degree=3)
    # if deterministic -> QuadratureInference (unscented gaussian approx)
    # if gaussian -> QuadratureGaussianInference (unscented gaussian approx)

    def sincos1(x):
        # input angles get split sin/cos
        x_sincos_shape = list(x.shape)
        x_sincos_shape[-1] += 1  # add another x dimension
        x_sincos = x.new_empty(x_sincos_shape, device=x.device)
        x_sincos[..., 0] = x[..., 0].sin()
        x_sincos[..., 1] = x[..., 0].cos()
        x_sincos[..., 2:] = x[..., 1:]
        return x_sincos

    class Input1SinCos(Decorator[Model]):
        @override
        def __call__(self, x, **kw):
            x_sincos = sincos1(x)
            return self.decorated.__call__(x_sincos, **kw)

    class InputUClamped(Decorator[Model]):
        @override
        def __call__(self, x, **kw):
            u_min, u_max = -environment.u_mx, +environment.u_mx
            x[..., dim_x] = torch.clamp(x[..., dim_x], u_min, u_max)
            res = self.decorated.__call__(x, **kw)
            if isinstance(res, Sequence):  # prevent nested tuples
                return *res, x
            else:
                return res, x

    class PredictDeltaX(Decorator[Model]):
        @override
        def __call__(self, x, **kw):
            delta, *rest = self.decorated.__call__(x, **kw)
            pred = x[..., :dim_x].detach() + delta
            return pred, *rest

    class WrapCpuDevice(Decorator[Model]):
        @override
        def __call__(self, x, **kw):
            res = self.decorated.__call__(x.cpu(), **kw)
            if isinstance(res, Sequence):  # prevent nested tuples
                return [ri.to(x.device) for ri in res]
            else:
                return res.to(x.device)

    #### ~ S: cost (model)
    cost = DeterministicCostModel(
        model=environment.cost,  # true cost
        approximate_inference=QuadratureImportanceSamplingInnovation(
            dim_xu,
            gh_params,
        ),
    )

    #### ~ S: global dynamics model
    if dyn_model_type == "env":
        global_dynamics = DeterministicDynamics(
            model=environment,
            approximate_inference=QuadratureInference(dim_xu, quad_params),
        )
        global_dynamics = WrapCpuDevice(global_dynamics)
    elif dyn_model_type == "mlp":
        global_dynamics = DeterministicDynamics(
            model=MultiLayerPerceptron(
                dim_xu + 1,  # Input1SinCos
                layer_spec_dyn,
                dim_x,
            ),
            approximate_inference=QuadratureInference(dim_xu, quad_params),
        )
        global_dynamics = PredictDeltaX(InputUClamped(Input1SinCos(global_dynamics)))
        # global_dynamics.model.init_whitening(dyn_train_buffer.xs, dyn_train_buffer.ys)
        loss_fn_dyn = lambda x, y: torch.nn.MSELoss()(global_dynamics.predict(x), y)
        opt_dyn = torch.optim.Adam(global_dynamics.model.parameters(), lr=lr_dyn)
    elif dyn_model_type == "resnet":
        global_dynamics = DeterministicDynamics(
            model=ResidualNetwork(
                dim_xu + 1,  # Input1SinCos
                layer_spec_dyn,
                dim_x,
            ),
            approximate_inference=QuadratureInference(dim_xu, quad_params),
        )
        global_dynamics = PredictDeltaX(InputUClamped(Input1SinCos(global_dynamics)))
        # global_dynamics.model.init_whitening(dyn_train_buffer.xs, dyn_train_buffer.ys)
        loss_fn_dyn = lambda x, y: torch.nn.MSELoss()(global_dynamics.predict(x), y)
        opt_dyn = torch.optim.Adam(global_dynamics.model.parameters(), lr=lr_dyn)
    elif dyn_model_type == "nlm-mlp":
        global_dynamics = StochasticDynamics(
            model=NeuralLinearModelMLP(
                dim_xu + 1,  # Input1SinCos
                layer_spec_dyn,
                n_features_dyn,
                dim_x,
            ),
            approximate_inference=QuadratureGaussianInference(dim_xu, quad_params),
        )
        global_dynamics = PredictDeltaX(InputUClamped(Input1SinCos(global_dynamics)))
        # global_dynamics.model.init_whitening(dyn_train_buffer.xs, dyn_train_buffer.ys)
        def loss_fn_dyn(xu, x_next):
            x, u = xu[..., :dim_x], xu[..., dim_x]
            u = torch.clamp(u, -environment.u_mx, +environment.u_mx)
            xu_sincos = sincos1(xu)
            delta_x = x_next - x
            return -global_dynamics.model.elbo(xu_sincos, delta_x)

        opt_dyn = torch.optim.Adam(global_dynamics.model.parameters(), lr=lr_dyn)
    elif dyn_model_type == "nlm-resnet":
        global_dynamics = StochasticDynamics(
            model=NeuralLinearModelResNet(
                dim_xu + 1,  # Input1SinCos
                layer_spec_dyn,
                n_features_dyn,
                dim_x,
            ),
            approximate_inference=QuadratureGaussianInference(dim_xu, quad_params),
        )
        global_dynamics = PredictDeltaX(InputUClamped(Input1SinCos(global_dynamics)))
        # global_dynamics.model.init_whitening(dyn_train_buffer.xs, dyn_train_buffer.ys)
        def loss_fn_dyn(xu, x_next):
            x, u = xu[..., :dim_x], xu[..., dim_x]
            u = torch.clamp(u, -environment.u_mx, +environment.u_mx)
            xu_sincos = sincos1(xu)
            delta_x = x_next - x
            return -global_dynamics.model.elbo(xu_sincos, delta_x)

        opt_dyn = torch.optim.Adam(global_dynamics.model.parameters(), lr=lr_dyn)
    elif dyn_model_type == "sngp":
        global_dynamics = StochasticDynamics(
            # global_dynamics = LBMDyn_SinCos1_Uclamp(
            model=SpectralNormalizedNeuralGaussianProcess(
                dim_xu + 1,  # Input1SinCos
                layer_spec_dyn,
                n_features_dyn,
                dim_x,
            ),
            approximate_inference=QuadratureGaussianInference(dim_xu, quad_params),
        )
        global_dynamics = PredictDeltaX(InputUClamped(Input1SinCos(global_dynamics)))
        # global_dynamics.model.init_whitening(dyn_train_buffer.xs, dyn_train_buffer.ys)
        def loss_fn_dyn(xu, x_next):
            x, u = xu[..., :dim_x], xu[..., dim_x]
            u = torch.clamp(u, -environment.u_mx, +environment.u_mx)
            xu_sincos = sincos1(xu)
            delta_x = x_next - x
            return -global_dynamics.model.elbo(xu_sincos, delta_x)

        opt_dyn = torch.optim.Adam(global_dynamics.model.parameters(), lr=lr_dyn)

    #### ~ S: local (i2c) policy
    local_policy = TimeVaryingStochasticPolicy(
        model=TimeVaryingLinearGaussian(
            horizon,
            dim_x,
            dim_u,
            action_covariance=0.2 * torch.eye(dim_u),
        ),
        approximate_inference=QuadratureGaussianInference(dim_x, quad_params),
    )

    def m_projection_loss(
        d_pred: MultivariateGaussian, d_target: MultivariateGaussian
    ) -> torch.Tensor:
        """
        m-projection (moment matching): D_KL[d_target || d_pred]
        when minimizing this w.r.t. d_target parameters (phi), the KL simplifies
        note: with Identity pred cov (=eye), the loss becomes MSE(mean_pred, mean_target)
        """
        mu_pred = d_pred.mean
        cov_pred = d_pred.covariance
        mu_target = d_target.mean
        cov_target = d_target.covariance
        # part1: trace(cov_pred**-1 @ cov_target)
        part1 = einops.einsum(
            torch.linalg.solve(cov_pred, cov_target), "... a a -> ..."
        )
        # part2: (mu_pred - mu_target).T @ cov_pred.inverse() @ (mu_pred - mu_target)
        mu_diff = mu_pred - mu_target
        part2 = einops.einsum(
            mu_diff, torch.linalg.solve(cov_pred, mu_diff), "... x, ... x -> ..."
        )
        part3 = cov_pred.logdet()
        return sum(part1 + part2 + part3)  # sum over all batch dims

    #### ~ S: global policy model
    if policy_type == "tvlg":
        global_policy = deepcopy(local_policy)
    elif policy_type == "mlp":
        global_policy = DeterministicPolicy(
            model=MultiLayerPerceptron(
                dim_x + 1,  # Input1SinCos
                layer_spec_pol,
                dim_u,
            ),
            approximate_inference=QuadratureInference(dim_x, quad_params),
        )
        global_policy = Input1SinCos(global_policy)
        # global_policy.model.init_whitening(train_buffer.xs, train_buffer.ys)
        opt_pol = torch.optim.Adam(global_policy.model.parameters(), lr=lr_pol)
    elif policy_type == "resnet":
        global_policy = DeterministicPolicy(
            model=ResidualNetwork(
                dim_x + 1,  # Input1SinCos
                layer_spec_pol,
                dim_u,
            ),
            approximate_inference=QuadratureInference(dim_x, quad_params),
        )
        global_policy = Input1SinCos(global_policy)
        # global_policy.model.init_whitening(train_buffer.xs, train_buffer.ys)
        opt_pol = torch.optim.Adam(global_policy.model.parameters(), lr=lr_pol)
    elif policy_type == "nlm-mlp":
        global_policy = StochasticPolicy(
            model=NeuralLinearModelMLP(
                dim_x + 1,  # Input1SinCos
                layer_spec_pol,
                n_features_pol,
                dim_u,
            ),
            approximate_inference=QuadratureGaussianInference(dim_x, quad_params),
        )
        global_policy = Input1SinCos(global_policy)
        # global_policy.model.init_whitening(pol_train_buffer.xs, pol_train_buffer.ys)
        opt_pol = torch.optim.Adam(global_policy.model.parameters(), lr=lr_pol)
    elif policy_type == "nlm-resnet":
        global_policy = StochasticPolicy(
            model=NeuralLinearModelResNet(
                dim_x + 1,  # Input1SinCos
                layer_spec_pol,
                n_features_pol,
                dim_u,
            ),
            approximate_inference=QuadratureGaussianInference(dim_x, quad_params),
        )
        global_policy = Input1SinCos(global_policy)
        # global_policy.model.init_whitening(pol_train_buffer.xs, pol_train_buffer.ys)
        opt_pol = torch.optim.Adam(global_policy.model.parameters(), lr=lr_pol)
    elif policy_type == "sngp":
        global_policy = StochasticPolicy(
            model=SpectralNormalizedNeuralGaussianProcess(
                dim_x + 1,  # Input1SinCos
                layer_spec_pol,
                n_features_pol,
                dim_u,
            ),
            approximate_inference=QuadratureGaussianInference(dim_x, quad_params),
        )
        global_policy = Input1SinCos(global_policy)
        # global_policy.model.init_whitening(pol_train_buffer.xs, pol_train_buffer.ys)
        opt_pol = torch.optim.Adam(global_policy.model.parameters(), lr=lr_pol)
    elif policy_type == "nlm_mlp_rff":
        global_policy = StochasticPolicy(
            model=NLM_MLP_RFF(
                dim_x + 1,  # Input1SinCos
                layer_spec_pol,
                n_features_pol,
                dim_u,
            ),
            approximate_inference=QuadratureInference(dim_x, quad_params),
        )
        global_policy = Input1SinCos(global_policy)
        # global_policy.model.init_whitening(train_buffer.xs, train_buffer.ys)
        opt_pol = torch.optim.Adam(global_policy.model.parameters(), lr=lr_pol)

    # all policies have same structure, so same loss
    def loss_fn_pol(s, a, **kw):
        return -global_policy.model.elbo(sincos1(s), a, **kw)

    #### ~ S: i2c solver
    i2c_solver = PseudoPosteriorSolver(
        dim_x=dim_x,
        dim_u=dim_u,
        horizon=horizon,
        dynamics=global_dynamics,
        cost=cost,
        policy_template=local_policy,
        # update_temperature_strategy=MaximumLikelihood(QuadratureInference(dim_xu, quad_params)),
        # update_temperature_strategy=QuadraticModel(QuadratureInference(dim_xu, quad_params)),
        # update_temperature_strategy=Constant(0.5 * torch.ones(n_i2c_vec, 1)),
        # update_temperature_strategy=Annealing(1e-2, 5, 35),
        # update_temperature_strategy=KullbackLeiblerDivergence(QuadratureInference(dim_xu, gh_params), epsilon=kl_bound),
        update_temperature_strategy=PolyakStepSize(
            QuadratureInference(dim_xu, quad_params)
        ),
    )

    ####################################################################################################################
    #### ~~~~~~ TRAINING ~~~~~~
    # fix seed again (different model setups make random numbers diverge)
    fix_random_seed(seed + 100)

    # TODO debug
    prefix = f"scripts/_dbg{n_i2c_vec}"
    # load (saved) policy
    state_dict = torch.load(
        repo_dir / f"{prefix}_pol_{policy_type}_ep{n_epochs_pol}.pth"
    )
    global_policy.model.load_state_dict(state_dict)

    dyn_epoch_counter = 0
    dyn_loss_trace = []
    dyn_test_loss_trace = []
    pol_epoch_counter = 0
    pol_loss_trace = []
    pol_test_loss_trace = []
    for i_iter in range(n_iter):
        logger.strong_line()
        logger.info(f"ITERATION {i_iter + 1}/{n_iter}")

        # # TODO debug
        # prefix = f"scripts/_dbg{n_i2c_vec}"
        # # load (saved) dynamics model
        # state_dict = torch.load(repo_dir / f"{prefix}_dyn_{dyn_model_type}_ep{n_epochs_dyn}.pth")
        # global_dynamics.model.load_state_dict(state_dict)

        # if False:
        if dyn_model_type != "env":
            #### ~ T: collect dynamics model data
            logger.weak_line()
            logger.info("START Collecting Dynamics Rollouts")
            global_dynamics.cpu()  # in-place
            global_dynamics.eval()
            torch.set_grad_enabled(False)

            class AddDithering(Decorator[Model]):  # decorate w/o changing policy
                @override
                def predict(self, x: torch.Tensor, **kw) -> torch.Tensor:
                    y = self.decorated.predict(x, **kw)
                    return y + 5e-2 * torch.randn_like(y)

            class RandomWalkPredict(Decorator[Model]):  # decorate w/o changing policy
                @override
                def predict(self, x: torch.Tensor, *, t, **kw) -> torch.Tensor:
                    if not hasattr(self, "y_current"):
                        y_ = self.decorated.predict(x, **kw)
                        self.y_current = torch.zeros_like(y_)
                        self.dy_current = torch.zeros_like(y_)
                    y_old = self.y_current
                    dy_old = self.dy_current
                    if t == 0:  # reset
                        y_old *= 0.0
                        dy_old *= 0.0
                    self.dy_current = dy_old + 0.1 * torch.randn_like(dy_old)
                    self.y_current = y_old + 5e-2 * self.dy_current
                    return self.y_current

            if i_iter == 0:  # untrained policy => use random noise (tvlg)
                exploration_policy = AddDithering(global_policy)
                # exploration_policy = RandomWalkPredict(global_policy)
            else:
                exploration_policy = AddDithering(global_policy)

            # train and test rollouts (env & exploration policy)
            for i in trange(n_dyn_train_episodes):
                state = initial_state_distribution.sample()
                s, a, ss = environment.run(state, exploration_policy, horizon)
                dyn_train_buffer.add([torch.hstack([s, a]), ss])
            n_dyn_test_episodes = max(1, int(n_dyn_train_episodes / train_test_ratio))
            for i in trange(n_dyn_test_episodes):
                state = initial_state_distribution.sample()
                s, a, ss = environment.run(state, exploration_policy, horizon)
                dyn_test_buffer.add([torch.hstack([s, a]), ss])
            logger.info("END Collecting Dynamics Rollouts")

            #### ~ T: train dynamics model
            logger.weak_line()
            logger.info("START Training Dynamics")
            global_dynamics.to(device)  # in-place
            global_dynamics.train()
            torch.set_grad_enabled(True)

            ## initial loss
            # for minibatch in dyn_train_buffer:  # TODO not whole buffer!
            #     _x, _y = minibatch
            #     _loss = loss_fn_dyn(_x, _y)
            #     dyn_loss_trace.append(_loss.detach().item())
            #     logger.log_data(
            #         **{
            #             "dynamics/train/loss": dyn_loss_trace[-1],
            #         },
            #     )
            # test_losses = []
            # for minibatch in dyn_test_buffer:  # TODO not whole buffer!
            #     _x_test, _y_test = minibatch
            #     _test_loss = loss_fn_dyn(_x_test, _y_test)
            #     test_losses.append(_test_loss.detach().item())
            # dyn_test_loss_trace.append(np.mean(test_losses))
            # logger.log_data(
            #     step=logger._step,  # in sync with training loss
            #     **{
            #         "dynamics/eval/loss": dyn_test_loss_trace[-1],
            #     },
            # )

            # torch.autograd.set_detect_anomaly(True)
            _train_losses = []
            for i_epoch_dyn in trange(n_epochs_dyn + 1):
                for i_minibatch, minibatch in enumerate(dyn_train_buffer):
                    sa, next_s = minibatch
                    opt_dyn.zero_grad()
                    loss = loss_fn_dyn(sa, next_s)
                    loss.backward()
                    opt_dyn.step()
                    dyn_loss_trace.append(loss.detach().item())
                    _train_losses.append(dyn_loss_trace[-1])
                    logger.log_data(
                        **{
                            # "dynamics/train/epoch": dyn_epoch_counter,
                            "dynamics/train/loss": dyn_loss_trace[-1],
                        },
                    )
                dyn_epoch_counter += 1

                # save logs & test
                logger.save_logs()
                if i_epoch_dyn % n_epochs_between_eval_dyn == 0:
                    with torch.no_grad():
                        # test loss
                        dyn_test_buffer.shuffling = False  # TODO only for plotting?
                        _test_losses = []
                        for minibatch in dyn_test_buffer:  # TODO not whole buffer!
                            _sa_test, _next_s_test = minibatch
                            _test_loss = loss_fn_dyn(_sa_test, _next_s_test)
                            _test_losses.append(_test_loss.item())
                        dyn_test_loss_trace.append(np.mean(_test_losses))
                        mean_train_loss = np.mean(_train_losses)
                        _train_losses = []
                        logger.log_data(
                            step=logger._step,  # in sync with training loss
                            **{
                                "dynamics/train/loss_mean": mean_train_loss,
                                "dynamics/eval/loss_mean": dyn_test_loss_trace[-1],
                            },
                        )
                        logstring = (
                            f"DYN: Epoch {i_epoch_dyn} ({dyn_epoch_counter}), "
                            f"Train Loss={mean_train_loss:.2}, "
                            f"Test Loss={dyn_test_loss_trace[-1]:.2}"
                        )
                        logger.info(logstring)

                # stop condition
                if i_epoch_dyn > min_epochs_per_train:
                    if (
                        dyn_test_loss_trace[-1] < early_stop_thresh
                        and dyn_loss_trace[-1] < early_stop_thresh
                    ):
                        break  # stop if test loss good

                # TODO save model more often than in each global iter?
                # if n % min(n_epochs * model_save_frequency, model_save_period) == 0:
                #     # Save the agent
                #     torch.save(model.state_dict(), results_dir / f"agent_{n}_{i_iter}.pth")

            # Save the model after training
            global_dynamics.cpu()  # in-place
            global_dynamics.eval()
            torch.set_grad_enabled(False)
            torch.save(
                global_dynamics.model.state_dict(),
                results_dir / f"dyn_model_{i_iter}.pth",
            )

            # TODO debug
            prefix = f"scripts/_dbg{n_i2c_vec}"
            # save dynamics model
            torch.save(
                global_dynamics.model.state_dict(),
                # repo_dir / f"{prefix}_dyn_rand_{dyn_model_type}_ep{n_epochs_dyn}.pth",
                repo_dir / f"{prefix}_dyn_pol_{dyn_model_type}_ep{n_epochs_dyn}.pth",
            )

            logger.info("END Training Dynamics")

            #### ~ T: plot dynamics model
            if plot_dyn and plotting:
                ## test dynamics model in rollouts
                # TODO extract?
                ## data traj (from buffer) -> trailing underscore = on device
                sa_env_ = dyn_test_buffer.data[0]  # [(n T) sa]
                # sa_env_ = dyn_train_buffer.data[0]  # [(n T) sa]
                ss_env = dyn_test_buffer.data[1].cpu()  # [(n T) sa]
                # ss_env = dyn_train_buffer.data[1].cpu()  # [(n T) sa]
                sa_env_ = einops.rearrange(sa_env_, "(n T) sa -> T n sa", T=horizon)
                ss_env = einops.rearrange(ss_env, "(n T) sa -> T n sa", T=horizon)
                # select how many trajs to plot (take from back of buffer)
                sa_env_ = sa_env_[:, -n_trajs_plot_dyn:, ...]
                ss_env = ss_env[:, -n_trajs_plot_dyn:, ...]
                _n_trajs = sa_env_.shape[1]
                sa_env = sa_env_.cpu()
                s_env, a_env = sa_env[..., :dim_x], sa_env[..., dim_x:]
                # pointwise
                ss_pred_pw_dists = []
                global_dynamics.to(device)
                _ss_pw_vec = global_dynamics.predict_dist(sa_env_[...]).cpu()
                global_dynamics.cpu()
                for t in range(horizon):
                    ss_pred_pw_dists.append(
                        MultivariateGaussian(
                            _ss_pw_vec.mean[t, ...], _ss_pw_vec.covariance[t, ...]
                        )
                    )
                ss_pred_pw = _ss_pw_vec.mean
                # rollout (replay action)
                ss_pred_roll_dists = []
                ss_pred_roll = torch.zeros_like(s_env)
                state = s_env[0, ...]  # for rollouts: data init state
                for t in range(horizon):
                    xu = torch.cat((state, a_env[t, ...]), dim=-1)
                    ss_pred_roll_dists.append(global_dynamics.predict_dist(xu))
                    ss_pred_roll[t, ...] = ss_pred_roll_dists[-1].mean
                    state = ss_pred_roll[t, ...]
                # compute costs (except init state use pred next state)
                c_env = cost.predict(sa_env)
                s_pred_pw = torch.cat([s_env[:1, ...], ss_pred_pw[:-1, ...]])
                sa_pred_pw = torch.cat([s_pred_pw, a_env], dim=-1)
                c_pw = cost.predict(sa_pred_pw)
                s_pred_roll = torch.cat([s_env[:1, ...], ss_pred_roll[:-1, ...]])
                sa_pred_roll = torch.cat([s_pred_roll, a_env], dim=-1)
                c_roll = cost.predict(sa_pred_roll)

                ### plot pointwise and rollout predictions (1 episode) ###
                fig, axs = plt.subplots(dim_u + 1 + dim_x, 2, **fig_kwargs)
                steps = torch.tensor(range(0, horizon))
                axs[0, 0].set_title("pointwise predictions")
                axs[0, 1].set_title("rollout predictions")
                for ti in range(_n_trajs):
                    # plot actions (twice: left & right)
                    for ui in range(dim_u):
                        axs[ui, 0].plot(steps, a_env[:, ti, ui], color="b")
                        axs[ui, 0].set_ylabel("action")
                        axs[ui, 1].plot(steps, a_env[:, ti, ui], color="b")
                        axs[ui, 1].set_ylabel("action")
                    # plot cost
                    ri = dim_u
                    axs[ri, 0].plot(steps, c_env[:, ti], color="b")
                    axs[ri, 0].plot(steps, c_pw[:, ti], color="r")
                    axs[ri, 0].set_ylabel("cost")
                    axs[ri, 1].plot(steps, c_env[:, ti], color="b")
                    axs[ri, 1].plot(steps, c_roll[:, ti], color="r")
                    axs[ri, 1].set_ylabel("cost")
                    for xi in range(dim_x):
                        xi_ = xi + dim_u + 1  # plotting offset
                        # plot pointwise state predictions
                        axs[xi_, 0].plot(steps, ss_env[:, ti, xi], color="b")
                        plot_mvn(
                            axs[xi_, 0], ss_pred_pw_dists, dim=xi, batch=ti, color="r"
                        )
                        axs[xi_, 0].set_ylabel(f"ss[{xi}]")
                        # plot rollout state predictions
                        axs[xi_, 1].plot(steps, ss_env[:, ti, xi], color="b")
                        plot_mvn(
                            axs[xi_, 1], ss_pred_roll_dists, dim=xi, batch=ti, color="r"
                        )
                        axs[xi_, 1].set_ylabel(f"ss[{xi}]")
                axs[-1, 0].set_xlabel("steps")
                axs[-1, 1].set_xlabel("steps")
                # replot with labels (prevent duplicate labels in legend)
                axs[ri, 0].plot(steps, c_env[:, ti], color="b", label="data")
                axs[ri, 0].plot(steps, c_pw[:, ti], color="r", label=dyn_model_type)
                add_grid(axs)
                handles, labels = axs[ri, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=2)
                fig.suptitle(
                    f"{dyn_model_type} pointwise and rollout dynamics on {_n_trajs} episode "
                    f"({int(dyn_train_buffer.size/horizon)} episodes, "
                    f"{dyn_epoch_counter} epochs, lr={lr_dyn:.0e})"
                )
                plt.savefig(results_dir / f"dyn_eval_{i_iter}.png")
                if show_plots:
                    plt.show()

            fig_loss_dyn, ax_loss_dyn = plt.subplots(**fig_kwargs)
            x_train_loss_dyn = scaled_xaxis(dyn_loss_trace, n_x_ticks=dyn_epoch_counter)
            ax_loss_dyn.plot(
                x_train_loss_dyn, dyn_loss_trace, c="k", label="train loss"
            )
            x_test_loss_dyn = scaled_xaxis(
                dyn_test_loss_trace, dist_x_ticks=n_epochs_between_eval_dyn
            )
            ax_loss_dyn.plot(
                x_test_loss_dyn, dyn_test_loss_trace, c="g", label="test loss"
            )
            if dyn_loss_trace[0] > 1 and dyn_loss_trace[-1] < 0.1:
                ax_loss_dyn.set_yscale("symlog")
            ax_loss_dyn.set_xlabel("epochs")
            ax_loss_dyn.set_ylabel("loss")
            add_grid(ax_loss_dyn)
            ax_loss_dyn.set_title(
                f"DYN {dyn_model_type} loss "
                f"({int(dyn_train_buffer.size/horizon)} episodes, lr={lr_dyn:.0e})"
            )
            ax_loss_dyn.legend()
            plt.savefig(results_dir / "dyn_loss.png")
            plt.show()

            ### policy vs dyn model
            if dyn_model_type != "env":
                xs = torch.zeros((horizon, dim_x))
                us = torch.zeros((horizon, dim_u))
                uvars = torch.zeros_like(us)
                xvars = torch.zeros_like(xs)
                state = initial_state_distribution.sample()
                for t in range(horizon):
                    action_dist = global_policy.predict_dist(state, t=t)
                    uvars[t, :] = action_dist.covariance.diagonal(dim1=-2, dim2=-1)
                    xu = torch.cat((state, action_dist.mean), dim=-1)[None, :]
                    nstate_dist = global_dynamics.predict_dist(xu)
                    xvars[t, :] = nstate_dist.covariance.diagonal(dim1=-2, dim2=-1)
                    xs[t, :] = state
                    us[t, :] = action_dist.mean
                    state = nstate_dist.mean[0, :]
                rollout_plot(
                    xs, us, u_max=u_max, uvars=uvars, xvars=xvars, **fig_kwargs
                )
                plt.suptitle(f"{policy_type} policy vs {dyn_model_type} dynamics")
                plt.savefig(
                    results_dir / f"{policy_type}_vs_{dyn_model_type}_{i_iter}.png"
                )
                plt.show()

        # return  # TODO DEBUG

        #### ~ T: i2c
        # i2c: find local (optimal) tvlg policy
        logger.weak_line()
        logger.info(f"START i2c [{n_iter_solver} iters]")
        # create initial state distributions for all i2c solutions (low inital velocity var)
        s0_vec_mean = initial_state_distribution.sample([n_i2c_vec])
        s0_i2c_cov = torch.diag_embed(torch.tensor([s0_i2c_var, s0dot_i2c_var]))
        s0_vec_cov = s0_i2c_cov.repeat(n_i2c_vec, 1, 1)
        s0_vec_dist = MultivariateGaussian(s0_vec_mean, s0_vec_cov)
        # learn a batch of local policies
        local_vec_policy, sa_posterior = i2c_solver(
            n_iteration=n_iter_solver,
            initial_state=s0_vec_dist,
            # initial_state=initial_state_distribution,  # TODO debug
            policy_prior=global_policy if i_iter != 0 else None,  # always start clean
            plot_posterior=plot_posterior and show_plots and plotting,
            delta_cost_thresh=delta_cost_thresh,
        )

        # # TODO debug
        # prefix = f"scripts/_dbg{n_i2c_vec}_"
        # # # save i2c
        # # torch.save(local_vec_policy, repo_dir / f'{prefix}local_vec_i2c.obj')
        # # torch.save(sa_posterior, repo_dir / f'{prefix}sa_posterior.obj')
        # # torch.save(i2c_solver.metrics, repo_dir / f'{prefix}i2c_solver-metrics.obj')
        # # torch.save(s0_vec_mean, repo_dir / f'{prefix}s0_vec_mean.obj')
        # # load (saved) i2c
        # local_vec_policy = torch.load(repo_dir / f"{prefix}local_vec_i2c.obj")
        # sa_posterior = torch.load(repo_dir / f"{prefix}sa_posterior.obj")
        # i2c_solver.metrics = torch.load(repo_dir / f"{prefix}i2c_solver-metrics.obj")
        # s0_vec_mean = torch.load(repo_dir / f"{prefix}s0_vec_mean.obj")
        # plot_trajectory_distribution(sa_posterior, "sa_posterior")

        logger.info("END i2c")
        # log i2c metrics
        for (k, v) in i2c_solver.metrics.items():
            for i_, vi in enumerate(v):
                log_dict = {"iter": i_iter, "i2c_iter": i_}
                log_dict[k] = vi  # one key, n_iter_solver values
            logger.log_data(log_dict)

        def mixture_tvlg(s, t):
            dist_vec = sa_posterior[t].to(s.device)
            # repeat state for each vec tvlg component
            s_roll_vec = einops.repeat(s, "b s -> b v s", v=n_i2c_vec)
            # get controller state distribution for t
            # (controller was optimized in local area around this)
            s_local_dist = dist_vec.marginalize(slice(0, dim_x))
            # compute importance weight: how close s_local_t is to s_roll
            eps = 1e-16  # prevent division by zero => when this is zero, numerator is also zero :)
            # s_local_dist.covariance *= 5  # higher spread
            importance_vec = s_local_dist.prob(s_roll_vec)
            # importance_vec = importance_vec.log().sigmoid()
            importance_sum = importance_vec.sum(dim=-2) + eps
            # get conditional action distribution
            # equiv. to tvlgc prediction, but with true conditional covariance
            a_cond_f = dist_vec.conditional(slice(dim_x, None))
            a_expert_dist = a_cond_f(s_roll_vec)
            # weight each prediction (mean & cov) by its importance ...
            a_expert_mean_weighted = a_expert_dist.mean * importance_vec
            a_expert_vars_weighted = einops.einsum(
                a_expert_dist.covariance,
                importance_vec,
                "... a a, ... a -> ... a",
            )
            # ... then sum and divide by sum of importances (normalize)
            a_expert_mean = a_expert_mean_weighted.sum(dim=-2) / (importance_sum)
            a_expert_vars = a_expert_vars_weighted.sum(dim=-2) / (importance_sum)
            a_expert_cov = torch.diag_embed(a_expert_vars + eps)  # prevent zero var
            importance = importance_sum / (a_expert_vars + eps)
            return MultivariateGaussian(a_expert_mean, a_expert_cov), importance

        #### ~ T: plot i2c local controller
        if plotting:
            ## plot (batch) i2c metrics
            fix, axs = plt.subplots(3, **fig_kwargs)
            temp_strategy_name = (
                i2c_solver.update_temperature_strategy.__class__.__name__
            )
            for i, (k, v) in enumerate(i2c_solver.metrics.items()):
                v = torch.stack(v)
                colors = plt.cm.brg(np.linspace(0, 1, n_i2c_vec))
                for b, c in zip(range(n_i2c_vec), colors):
                    axs[i].plot(v[:, b], color=c)
                axs[i].set_ylabel(k)
            add_grid(axs)
            plt.suptitle(
                f"{n_i2c_vec} i2c metrics (temp.strategy: {temp_strategy_name})"
            )
            plt.savefig(results_dir / f"i2c_metrics_{i_iter}.png")

            ## plot local policies vs env
            xs, us, xxs = environment.run(s0_vec_mean, local_vec_policy, horizon)
            uvars = []
            for t in range(horizon):
                u_dist = local_vec_policy.predict_dist(xs[t, ...], t=t)
                u_var = u_dist.covariance.diagonal(dim1=-2, dim2=-1)
                uvars.append(u_var)
            uvars = torch.stack(uvars)
            rollout_plot(xs, us, uvars=uvars, u_max=u_max, **fig_kwargs)
            plt.suptitle(f"{n_i2c_vec} local policies vs env")
            plt.savefig(results_dir / f"{n_i2c_vec}_tvlgs_vs_env_{i_iter}.png")

            ## tvlg vec vs dyn model
            xs = torch.zeros((horizon, n_i2c_vec, dim_x))
            xvars = torch.zeros((horizon, n_i2c_vec, dim_x))
            us = torch.zeros((horizon, n_i2c_vec, dim_u))
            uvars = torch.zeros((horizon, n_i2c_vec, dim_u))
            s_dist = MultivariateGaussian.from_deterministic(s0_vec_mean)
            for t in range(horizon):
                xs[t, ...] = s_dist.mean
                xvars[t, ...] = s_dist.covariance.diagonal(dim1=-2, dim2=-1)
                a_dist = local_vec_policy.predict_dist(xs[t, ...], t=t)
                us[t, ...] = a_dist.mean
                uvars[t, ...] = a_dist.covariance.diagonal(dim1=-2, dim2=-1)
                xu = torch.cat((xs[t, ...], us[t, ...]), dim=-1)
                s_dist = global_dynamics.predict_dist(xu)
            # env.plot does not use env, it only plots
            rollout_plot(xs, us, xvars=xvars, uvars=uvars, u_max=u_max, **fig_kwargs)
            plt.suptitle(f"{n_i2c_vec} local policies vs {dyn_model_type} dynamics")
            plt.savefig(
                results_dir / f"{n_i2c_vec}_tvlgs_vs_{dyn_model_type}-dyn_{i_iter}.png",
            )

            ## plot mixture policy vs env
            xs = torch.zeros((horizon, n_i2c_vec, dim_x))
            xvars = torch.zeros((horizon, n_i2c_vec, dim_x))
            us = torch.zeros((horizon, n_i2c_vec, dim_u))
            uvars = torch.zeros((horizon, n_i2c_vec, dim_u))
            s_dist = MultivariateGaussian.from_deterministic(s0_vec_mean)
            for t in range(horizon):
                xs[t, ...] = s_dist.mean
                xvars[t, ...] = s_dist.covariance.diagonal(dim1=-2, dim2=-1)
                a_dist, importance = mixture_tvlg(xs[t, ...], t=t)
                us[t, ...] = a_dist.mean
                uvars[t, ...] = a_dist.covariance.diagonal(dim1=-2, dim2=-1)
                xu = torch.cat((xs[t, ...], us[t, ...]), dim=-1)
                s_dist = global_dynamics.predict_dist(xu)
            # env.plot does not use env, it only plots
            rollout_plot(xs, us, xvars=xvars, uvars=uvars, u_max=u_max, **fig_kwargs)
            plt.suptitle(f"mixture policy vs env")
            plt.savefig(results_dir / f"mixture_tvlg_vs_env_{i_iter}.png")

            if plot_local_policy and show_plots:
                plt.show()

        return  # TODO DEBUG

        # ### plot data space coverage ###
        # if plot_data:
        #     cols = ["theta", "theta_dot"]  # TODO useful to also plot actions?
        #     # train data
        #     states = pol_train_buffer.data[0]
        #     df = pd.DataFrame()
        #     df[cols] = np.array(states.cpu())
        #     df["traj_id"] = df.index // horizon
        #     g = sns.PairGrid(df, hue="traj_id")
        #     # g = sns.PairGrid(df)
        #     g.map_diag(sns.histplot, hue=None)
        #     g.map_offdiag(plt.plot)
        #     g.fig.suptitle(f"train data ({df.shape[0] // horizon} episodes)", y=1.01)
        #     g.savefig(results_dir / "train_data.png", dpi=150)
        #     # test data
        #     states = pol_test_buffer.data[0]
        #     df = pd.DataFrame()
        #     df[cols] = np.array(states.cpu())
        #     df["traj_id"] = df.index // horizon
        #     g = sns.PairGrid(df, hue="traj_id")
        #     # g = sns.PairGrid(df)
        #     g.map_diag(sns.histplot, hue=None)
        #     g.map_offdiag(plt.plot)
        #     g.fig.suptitle(f"test data ({df.shape[0] // horizon} episodes)", y=1.01)
        #     g.savefig(results_dir / "test_data.png", dpi=150)

        if policy_type == "tvlg":
            # create mixture (mean) policy
            global_policy = deepcopy(local_vec_policy)
            # # a) average gains
            # global_policy.model.k_actual = local_vec_policy.model.k_actual.mean(1)
            # global_policy.model.K_actual = local_vec_policy.model.K_actual.mean(1)
            # global_policy.model.k_opt = local_vec_policy.model.k_opt.mean(1)
            # global_policy.model.K_opt = local_vec_policy.model.K_opt.mean(1)
            # global_policy.model.K_opt = local_vec_policy.model.sigma.mean(1)
            # global_policy.model.K_opt = local_vec_policy.model.chol.mean(1)
            # b) take first gains
            global_policy.model.k_actual = local_vec_policy.model.k_actual[:, 0, ...]
            global_policy.model.K_actual = local_vec_policy.model.K_actual[:, 0, ...]
            global_policy.model.k_opt = local_vec_policy.model.k_opt[:, 0, ...]
            global_policy.model.K_opt = local_vec_policy.model.K_opt[:, 0, ...]
            global_policy.model.sigma = local_vec_policy.model.sigma[:, 0, ...]
            global_policy.model.chol = local_vec_policy.model.chol[:, 0, ...]
        else:
            #### ~ T: collect policy data
            logger.weak_line()
            logger.info("START Collecting Policy Rollouts")
            global_dynamics.cpu()  # in-place
            global_dynamics.eval()
            torch.set_grad_enabled(False)

            # Fit global policy to local policy
            def dagger_rollout(s_init, p_expert=1.0, stoch=False, var_factor=1.0):
                s_roll = torch.empty([horizon, n_i2c_vec, dim_x]).to(s_init.device)
                a_means = torch.empty([horizon, n_i2c_vec, dim_u]).to(s_init.device)
                a_covs = torch.empty([horizon, n_i2c_vec, dim_u, dim_u]).to(
                    s_init.device
                )
                importances = torch.empty([horizon, n_i2c_vec, 1]).to(s_init.device)
                s_current = s_init
                for t in range(horizon):
                    a_expert_dist, importance = mixture_tvlg(s_current, t)
                    # store expert prediction data for buffer
                    s_roll[t, ...] = s_current
                    a_means[t, ...] = a_expert_dist.mean
                    a_covs[t, ...] = a_expert_dist.covariance
                    importances[t, ...] = importance
                    # asdf = torch.rand(1)
                    # s = s': p% expert vs policy step
                    # if torch.rand(1) < p_expert:
                    #     if stoch:
                    #         a_expert_dist.covariance /= 10
                    #         a_step = a_expert_dist.sample()
                    #     else:
                    #         a_step = a_expert_dist.mean
                    # else:
                    #     # predict 1 step with trained policy & dynamics model step => (s, a)
                    #     a_step = global_policy.predict(s_current)
                    a_step = a_expert_dist.mean
                    # dynamics: predict next state (no policy uncertainty?)
                    sa = torch.cat((s_current, a_step), dim=-1)
                    s_current = global_dynamics.predict(sa)
                # for b in range(n_i2c_vec):
                #     rollout_plot(s_roll[:, b, :], a_means[:, b, :], uvars=a_covs[:, b, :], u_max=u_max, **fig_kwargs)
                # split train/test (split different i2c trajs)
                n_train = max(
                    1, int(n_i2c_vec * train_test_ratio / (train_test_ratio + 1))
                )
                train_test_data = []
                for indices in [slice(0, n_train), slice(n_train, None)]:
                    # add batched rollouts sequentially (not interleaved) => time dim last batch dim
                    s_roll_ = einops.rearrange(
                        s_roll[:, indices, :], "T ... s -> (... T) s"
                    )
                    a_means_ = einops.rearrange(
                        a_means[:, indices, :], "T ... a -> (... T) a"
                    )
                    a_covs_ = einops.rearrange(
                        a_covs[:, indices, :], "T ... a1 a2 -> (... T) a1 a2"
                    )
                    importances_ = einops.rearrange(
                        importances[:, indices, :], "T ... one -> (... T) one"
                    )
                    train_test_data.append([s_roll_, a_means_, a_covs_, importances_])
                return train_test_data

            # collect train & test data
            if pol_strategy == "moment matching":
                # store marginal state and conditional action of joint (smoothed) posterior
                # (state covariance not needed for moment matching: policy has no cov input)
                # sample multiple values using quadrature (feedback behaviour)
                quad_s = QuadratureInference(dim_x, quad_params)
                n_samples = quad_s.n_points
                s_pts = torch.empty([horizon, n_samples, n_i2c_vec, dim_x])
                a_means = torch.empty([horizon, n_samples, n_i2c_vec, dim_u])
                a_covs = torch.empty([horizon, n_samples, n_i2c_vec, dim_u, dim_u])
                for t, dist_vec_t in enumerate(sa_posterior):
                    s_dist = dist_vec_t.marginalize(slice(0, dim_x))
                    s_pts[t, ...] = einops.rearrange(
                        quad_s.get_x_pts(s_dist.mean, s_dist.covariance),
                        "v b s -> b v s",
                    )
                    a_dist_f = dist_vec_t.conditional(slice(dim_x, None))
                    a_dist = a_dist_f(s_pts[t, ...])
                    a_means[t, ...] = a_dist.mean
                    a_covs[t, ...] = a_dist.covariance
                # split train/test (split different i2c trajs)
                n_train = max(
                    1, int(n_i2c_vec * train_test_ratio / (train_test_ratio + 1))
                )
                for buf, indices in [
                    (pol_train_buffer, slice(0, n_train)),
                    (pol_test_buffer, slice(n_train, None)),
                ]:
                    # add local solutions sequentially (not interleaved) => time dim last batch dim
                    s_pts = einops.rearrange(s_pts, "T ... s -> (... T) s")
                    a_means = einops.rearrange(a_means, "T ... a -> (... T) a")
                    a_covs = einops.rearrange(a_covs, "T ... a1 a2 -> (... T) a1 a2")
                    importances = torch.ones_like(s_pts[..., 0:1])  # equal importance
                    buf.clear()  # only train policy on last controller
                    buf.add([s_pts, a_means, a_covs, importances])
            elif pol_strategy == "dagger":
                s_init = s0_vec_mean.to(device)
                train_data, test_data = dagger_rollout(s_init)
                pol_train_buffer.add(train_data)
                pol_test_buffer.add(test_data)

            #### ~ T: train policy
            logger.weak_line()
            logger.info("START Training Policy")
            global_policy.to(device)  # in-place
            global_dynamics.to(device)  # in-place
            torch.set_grad_enabled(True)

            _train_losses = []
            for i_epoch_pol in trange(n_epochs_pol + 1):
                # for every epoch fit train data
                for i_minibatch, minibatch in enumerate(pol_train_buffer):
                    s_mean, a_mean, a_cov, importance = minibatch
                    opt_pol.zero_grad()
                    if pol_strategy == "moment matching":
                        if isinstance(global_policy, StochasticPolicy):
                            a_pred_dist = global_policy.predict_dist(s_mean)
                            a_dist = MultivariateGaussian(a_mean, a_cov)
                            loss = m_projection_loss(a_pred_dist, a_dist)
                        else:
                            a_pred_mean = global_policy.predict(s_mean)
                            loss = torch.nn.MSELoss()(a_pred_mean, a_mean)
                    elif pol_strategy == "dagger":
                        # importance = importance.log().sigmoid()
                        # loss = loss_fn_pol(s_mean, a_mean, loss_factor=importance)
                        loss = loss_fn_pol(s_mean, a_mean)
                    loss.backward()
                    opt_pol.step()

                    pol_loss_trace.append(loss.detach().item())
                    _train_losses.append(pol_loss_trace[-1])
                    logger.log_data(
                        **{
                            # "policy/train/epoch": pol_epoch_counter,
                            "policy/train/loss": pol_loss_trace[-1],
                        },
                    )

                if i_epoch_pol % n_epochs_between_rollouts == 0:
                    with torch.no_grad():
                        if pol_strategy == "dagger":
                            # dagger rollouts
                            s_init = s0_vec_mean.to(device)
                            train_data, test_data = dagger_rollout(s_init)
                            pol_train_buffer.add(train_data)
                            pol_test_buffer.add(test_data)

                def visualize_training():
                    if not plotting:
                        return
                    fig, axs = plt.subplots(3, **fig_kwargs)
                    s_mean, a_mean, a_cov, importance = pol_train_buffer.data  # sorted
                    a_cov_diag = einops.einsum(a_cov, "... x x -> ... x")
                    s_mean = einops.rearrange(s_mean, "(n T) s -> n T s", T=horizon)
                    a_mean = einops.rearrange(a_mean, "(n T) a -> n T a", T=horizon)
                    a_cov_diag = einops.rearrange(
                        a_cov_diag, "(n T) a -> n T a", T=horizon
                    )
                    importance = einops.rearrange(
                        importance, "(n T) one -> n T one", T=horizon
                    )
                    if isinstance(global_policy, StochasticPolicy):
                        a_pred_dist = global_policy.predict_dist(s_mean)
                        a_pred_mean = a_pred_dist.mean
                        a_pred_cov_diag = einops.einsum(
                            a_pred_dist.covariance, "... x x -> ... x"
                        )
                    else:
                        a_pred_mean = global_policy.predict(s_mean)
                    n = s_mean.shape[0]
                    for i in range(n):
                        axs[0].plot(a_mean[i, ...].detach().cpu(), label="data", c="C0")
                        axs[0].plot(
                            a_pred_mean[i, ...].detach().cpu(), label="pred", c="C1"
                        )
                        axs[1].plot(
                            a_cov_diag[i, ...].detach().cpu(), label="data", c="C0"
                        )
                        if isinstance(global_policy, StochasticPolicy):
                            axs[1].plot(
                                a_pred_cov_diag[i, ...].detach().cpu(),
                                label="pred",
                                c="C1",
                            )
                        axs[2].semilogy(
                            importance[i, ...].detach().cpu(),
                            label="importance",
                            c="C2",
                        )

                    axs[0].set_ylabel("mean")
                    axs[1].set_ylabel("variance")
                    axs[2].set_ylabel("importance")
                    add_grid(axs)
                    handles, labels = axs[0].get_legend_handles_labels()
                    fig.legend(handles[:2], labels[:2], loc="lower center", ncol=2)
                    # axs[0].legend()
                    fig.suptitle(
                        f"{policy_type} distilling {n} i2c solutions (epoch {pol_epoch_counter})"
                    )
                    plt.show()
                    # bias = global_policy.model.pred_var_bias().detach().cpu().item()
                    # error = global_policy.model.error_cov_out().detach().cpu().item()
                    # logger.info(f"bias={bias}, error={error}")

                pol_epoch_counter += 1

                # save logs & test
                logger.save_logs()
                if i_epoch_pol % n_epochs_between_eval_pol == 0:
                    with torch.no_grad():
                        # visualize_training()
                        # test loss
                        pol_test_buffer.shuffling = False  # TODO only for plotting?
                        _test_losses = []
                        for minibatch in pol_test_buffer:  # TODO not whole buffer!
                            _s_test, _a_test, _a_cov_test, _importance = minibatch
                            if pol_strategy == "moment matching":
                                if isinstance(global_policy, StochasticPolicy):
                                    _a_pred_dist = global_policy.predict_dist(_s_test)
                                    _a_test_dist = MultivariateGaussian(
                                        _a_test, _a_cov_test
                                    )
                                    _test_loss = m_projection_loss(
                                        _a_pred_dist, _a_test_dist
                                    )
                                else:
                                    _a_pred = global_policy.predict(_s_test)
                                    _test_loss = torch.nn.MSELoss()(_a_pred, _a_test)
                            elif pol_strategy == "dagger":
                                # _importance = _importance.log().sigmoid()
                                # _test_loss = loss_fn_pol(_s_test, _a_test, loss_factor=_importance)
                                _test_loss = loss_fn_pol(_s_test, _a_test)
                            _test_losses.append(_test_loss.item())
                        pol_test_loss_trace.append(np.mean(_test_losses))
                        mean_train_loss = np.mean(_train_losses)
                        _train_losses = []
                        logger.log_data(
                            step=logger._step,  # in sync with training loss
                            **{
                                "policy/train/loss_mean": mean_train_loss,
                                # "policy/eval/loss": pol_test_loss_trace[-1],
                            },
                        )
                        logstring = (
                            f"POL: Epoch {i_epoch_pol} ({pol_epoch_counter}), "
                            f"Train Loss={mean_train_loss:.2}, "
                            f"Test Loss={pol_test_loss_trace[-1]:.2}"
                        )
                        logger.info(logstring)

                # stop condition
                if i_epoch_pol > min_epochs_per_train:
                    if (
                        pol_test_loss_trace[-1] < early_stop_thresh
                        and pol_loss_trace[-1] < early_stop_thresh
                    ):
                        break  # stop if test loss good

                # TODO save model more often than in each global iter?
                # if n % min(n_epochs * model_save_frequency, model_save_period) == 0:
                #     # Save the agent
                #     torch.save(model.state_dict(), results_dir / f"agent_{n}_{i_iter}.pth")

            # Save the model after training
            global_policy.cpu()  # in-place
            global_policy.eval()
            torch.set_grad_enabled(False)
            torch.save(
                global_policy.model.state_dict(),
                results_dir / f"pol_model_{i_iter}.pth",
            )
            logger.info("END Training policy")

            # # TODO debug
            # prefix = f"scripts/_dbg{n_i2c_vec}"
            # # save policy
            # torch.save(
            #     global_policy.model.state_dict(),
            #     repo_dir / f"{prefix}_pol_{policy_type}_ep{n_epochs_pol}.pth",
            # )

            #### ~ T: plot policy
            if plot_policy and plotting:
                # compute jacobian of policy action & compare with tvlg K_actual
                states = pol_test_buffer.data[0]  # [(b T) x]
                # states = pol_train_buffer.data[0]  # [(b T) x]
                global_policy.to(device)
                jac_f = vmap(jacrev(global_policy.predict))
                jac = jac_f(states)
                jac = einops.rearrange(jac, "(b T) y x -> T b y x", T=horizon)
                jac = jac.cpu()
                global_policy.cpu()
                torch.cuda.empty_cache()
                K = local_vec_policy.model.K_actual  # (T b y x)

                fig, axs = plt.subplots(dim_x, 1, **fig_kwargs)
                yi = 0  # dim_y == 1
                for xi in range(dim_x):
                    for bi in range(jac.shape[1]):
                        axs[xi].plot(jac[:, bi, yi, xi], c="C1")
                    # add label
                    axs[xi].plot(jac[:, bi, yi, xi], c="C1", label="policy jac")
                    for bi in range(K.shape[1]):
                        axs[xi].plot(K[:, bi, yi, xi], c="C0")
                    # add label
                    axs[xi].plot(K[:, bi, yi, xi], c="C0", label="tvlc K")
                    axs[xi].set_ylabel(f"da/ds_{xi}")
                axs[xi].set_xlabel("steps")
                add_grid(axs)
                handles, labels = axs[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=2)
                fig.suptitle(
                    f"{policy_type} action jacobian vs tvlg K "
                    f"({int(pol_train_buffer.size/horizon)} episodes, "
                    f"{pol_epoch_counter} epochs, lr={lr_pol:.0e})"
                )
                plt.savefig(
                    results_dir / f"{policy_type}_jac_vs_tvlg_K_{i_iter}.png",
                )

                ## test policy in rollouts
                # TODO extract?
                ## data traj (from buffer) -> trailing underscore = on device
                s_env_ = pol_test_buffer.data[0]  # [(n T) s]
                a_env = pol_test_buffer.data[1].cpu()  # [(n T) a]
                # s_env_ = pol_train_buffer.data[0]  # [(n T) s]
                # a_env = pol_train_buffer.data[1].cpu()  # [(n T) a]
                s_env_ = einops.rearrange(s_env_, "(n T) s -> T n s", T=horizon)
                a_env = einops.rearrange(a_env, "(n T) a -> T n a", T=horizon)
                # select how many trajs to plot (take from back of buffer)
                a_env = a_env[:, -n_trajs_plot_pol:, ...]
                s_env_ = s_env_[:, -n_trajs_plot_pol:, ...]
                _n_trajs = s_env_.shape[1]
                s_env = s_env_.cpu()
                # pointwise
                a_pred_pw_dists = []
                global_policy.to(device)
                _a_pw_vec = global_policy.predict_dist(s_env_[...]).cpu()
                global_policy.cpu()
                for t in range(horizon):
                    a_pred_pw_dists.append(
                        MultivariateGaussian(
                            _a_pw_vec.mean[t, ...], _a_pw_vec.covariance[t, ...]
                        )
                    )
                a_pred_pw = _a_pw_vec.mean
                # rollout
                a_pred_roll_dists = []
                a_pred_roll = torch.zeros_like(a_env)
                s_pred_roll = torch.zeros_like(s_env)
                state = s_env[0, ...]  # for rollouts: data init state
                for t in range(horizon):
                    s_pred_roll[t, ...] = state
                    a_pred_roll_dists.append(global_policy.predict_dist(state))
                    a_pred_roll[t, ...] = a_pred_roll_dists[-1].mean
                    # next state
                    xu = torch.cat([state, a_pred_roll[t, ...]], dim=-1)
                    state, _ = environment(xu)
                # compute costs (except init state use pred next state)
                sa_env = torch.cat([s_env, a_env], dim=-1)
                c_env = cost.predict(sa_env)
                sa_pred_pw = torch.cat([s_env, a_pred_pw], dim=-1)
                c_pw = cost.predict(sa_pred_pw)
                sa_pred_roll = torch.cat([s_pred_roll, a_pred_roll], dim=-1)
                c_roll = cost.predict(sa_pred_roll)

                ### plot pointwise and rollout predictions (1 episode) ###
                fig, axs = plt.subplots(dim_u + 1 + dim_x, 2, **fig_kwargs)
                steps = torch.tensor(range(0, horizon))
                axs[0, 0].set_title("pointwise predictions")
                axs[0, 1].set_title("rollout predictions")
                for ti in range(_n_trajs):
                    # plot actions (twice: left & right)
                    for ui in range(dim_u):
                        axs[ui, 0].plot(steps, a_env[:, ti, ui], color="b")
                        plot_mvn(
                            axs[ui, 0], a_pred_pw_dists, dim=ui, batch=ti, color="r"
                        )
                        axs[ui, 0].set_ylabel("action")
                        axs[ui, 1].plot(steps, a_env[:, ti, ui], color="b")
                        plot_mvn(
                            axs[ui, 1], a_pred_roll_dists, dim=ui, batch=ti, color="r"
                        )
                        axs[ui, 1].set_ylabel("action")
                    # plot cost
                    ri = dim_u
                    axs[ri, 0].plot(steps, c_env[:, ti], color="b")
                    axs[ri, 0].plot(steps, c_pw[:, ti], color="r")
                    axs[ri, 0].set_ylabel("cost")
                    axs[ri, 1].plot(steps, c_env[:, ti], color="b")
                    axs[ri, 1].plot(steps, c_roll[:, ti], color="r")
                    axs[ri, 1].set_ylabel("cost")
                    for xi in range(dim_x):
                        xi_ = xi + dim_u + 1  # plotting offset
                        # plot pointwise state predictions
                        axs[xi_, 0].plot(steps, s_env[:, ti, xi], color="b")
                        axs[xi_, 0].set_ylabel(f"s[{xi}]")
                        # plot rollout state predictions
                        axs[xi_, 1].plot(steps, s_env[:, ti, xi], color="b")
                        axs[xi_, 1].plot(steps, s_pred_roll[:, ti, xi], color="r")
                        axs[xi_, 1].set_ylabel(f"s[{xi}]")
                axs[-1, 0].set_xlabel("steps")
                axs[-1, 1].set_xlabel("steps")
                # replot with labels (prevent duplicate labels in legend)
                axs[ri, 0].plot(steps, c_env[:, ti], color="b", label="i2c")
                axs[ri, 0].plot(steps, c_pw[:, ti], color="r", label=policy_type)
                add_grid(axs)
                handles, labels = axs[ri, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=2)
                fig.suptitle(
                    f"{policy_type} pointwise and rollout policy on {_n_trajs} episode "
                    f"({int(pol_train_buffer.size/horizon)} episodes, "
                    f"{pol_epoch_counter} epochs, lr={lr_pol:.0e})"
                )
                plt.savefig(results_dir / f"pol_eval_{i_iter}.png")

                # plot policy rollouts
                # a) from i2c mixture starting positions
                xs, us, xxs = environment.run(s0_vec_mean, global_policy, horizon)
                uvars = []
                for t in range(horizon):
                    u_dist = global_policy.predict_dist(xs[t, ...], t=t)
                    u_var = u_dist.covariance.diagonal(dim1=-2, dim2=-1)
                    uvars.append(u_var)
                uvars = torch.stack(uvars)
                rollout_plot(xs, us, uvars=uvars, u_max=u_max, **fig_kwargs)
                plt.suptitle(f"{policy_type} policy vs env (from i2c init state)")
                plt.savefig(
                    results_dir / f"{policy_type}_vs_env_from_s0i2c_{i_iter}.png",
                )
                # # b) from env starting position
                # initial_state = initial_state_distribution.sample()
                # xs, us, xxs = environment.run(initial_state, global_policy, horizon)
                # uvars = torch.zeros_like(us)
                # for t in range(horizon):
                #     u_dist = global_policy.predict_dist(xs[t, ...], t=t)
                #     uvar[t, :] = u_dist.covariance.diagonal(dim1=-2, dim2=-1)
                # rollout_plot(xs, us, uvars=uvars, u_max=u_max, **fig_kwargs)
                # plt.suptitle(f"{policy_type} policy vs env")
                # plt.savefig(results_dir / f"{policy_type}_vs_env_{i_iter}.png")

                if show_plots:
                    plt.show()

            fig_loss_pol, ax_loss_pol = plt.subplots(**fig_kwargs)
            x_train_loss_pol = scaled_xaxis(pol_loss_trace, n_x_ticks=pol_epoch_counter)
            ax_loss_pol.plot(
                x_train_loss_pol, pol_loss_trace, c="k", label="train loss"
            )
            x_test_loss_pol = scaled_xaxis(
                pol_test_loss_trace, dist_x_ticks=n_epochs_between_eval_pol
            )
            ax_loss_pol.plot(
                x_test_loss_pol, pol_test_loss_trace, c="g", label="test loss"
            )
            if pol_loss_trace[0] > 1 and pol_loss_trace[-1] < 0.1:
                ax_loss_pol.set_yscale("symlog")
            ax_loss_pol.set_xlabel("epochs")
            ax_loss_pol.set_ylabel("loss")
            add_grid(ax_loss_pol)
            ax_loss_pol.set_title(
                f"POL {policy_type} loss "
                f"({int(pol_train_buffer.size/horizon)} local solutions, lr={lr_pol:.0e})"
            )
            ax_loss_pol.legend()
            plt.savefig(results_dir / "pol_loss.png")

        # return  # TODO DEBUG

    ####################################################################################################################
    #### ~~~~~~ EVALUATION ~~~~~~

    # local_policy.plot_metrics()
    # initial_state_distribution.mean = torch.Tensor([torch.pi + 0.4, 0.0])  # breaks local_policy!!
    initial_state = initial_state_distribution.sample([1])

    if plotting:
        ### policy vs env
        xs, us, xxs = environment.run(initial_state, global_policy, horizon)
        uvars = torch.zeros_like(us)
        for t in range(horizon):
            u_dist = global_policy.predict_dist(xs[t, ...], t=t)
            uvars[t, :] = u_dist.covariance.diagonal(dim1=-2, dim2=-1)
        rollout_plot(xs, us, uvars=uvars, u_max=u_max, **fig_kwargs)
        plt.suptitle(f"{policy_type} policy vs env")
        plt.savefig(results_dir / f"{policy_type}_vs_env_{i_iter}.png")

        ### policy vs dyn model
        if dyn_model_type != "env":
            xs = torch.zeros((horizon, dim_x))
            us = torch.zeros((horizon, dim_u))
            xvars = torch.zeros_like(xs)
            uvars = torch.zeros_like(us)
            state = initial_state
            for t in range(horizon):
                action_dist = global_policy.predict_dist(state, t=t)
                uvars[t, :] = action_dist.covariance.diagonal(dim1=-2, dim2=-1)
                xu = torch.cat((state, action_dist.mean), dim=-1)[None, :]
                nstate_dist = global_dynamics.predict_dist(xu)
                xvars[t, :] = action_dist.covariance.diagonal(dim1=-2, dim2=-1)
                xs[t, :] = state
                us[t, :] = action_dist.mean
                state = nstate_dist.mean[0, :]
            rollout_plot(xs, us, uvars=uvars, xvars=xvars, u_max=u_max, **fig_kwargs)
            plt.suptitle(f"{policy_type} policy vs {dyn_model_type} dynamics")
            plt.savefig(results_dir / f"{policy_type}_vs_{dyn_model_type}_{i_iter}.png")

        ### plot data space coverage ###
        if plot_data and plotting:
            cols = ["theta", "theta_dot"]  # TODO useful to also plot actions?
            # train data
            xs = dyn_train_buffer.data[0][:, :dim_x]  # only states
            df = pd.DataFrame()
            df[cols] = np.array(xs.cpu())
            df["traj_id"] = df.index // horizon
            g = sns.PairGrid(df, hue="traj_id")
            # g = sns.PairGrid(df)
            g.map_diag(sns.histplot, hue=None)
            g.map_offdiag(plt.plot)
            g.fig.suptitle(f"train data ({df.shape[0] // horizon} episodes)", y=1.01)
            g.savefig(results_dir / "train_data.png", dpi=150)
            # test data
            xs = dyn_test_buffer.data[0][:, :dim_x]  # only states
            df = pd.DataFrame()
            df[cols] = np.array(xs.cpu())
            df["traj_id"] = df.index // horizon
            g = sns.PairGrid(df, hue="traj_id")
            # g = sns.PairGrid(df)
            g.map_diag(sns.histplot, hue=None)
            g.map_offdiag(plt.plot)
            g.fig.suptitle(f"test data ({df.shape[0] // horizon} episodes)", y=1.01)
            g.savefig(results_dir / "test_data.png", dpi=150)

        # plot training loss
        if dyn_model_type != "env":
            fig_loss_dyn, ax_loss_dyn = plt.subplots(**fig_kwargs)
            x_train_loss_dyn = scaled_xaxis(dyn_loss_trace, n_x_ticks=dyn_epoch_counter)
            ax_loss_dyn.plot(
                x_train_loss_dyn, dyn_loss_trace, c="k", label="train loss"
            )
            x_test_loss_dyn = scaled_xaxis(
                dyn_test_loss_trace, dist_x_ticks=n_epochs_between_eval_pol
            )
            ax_loss_dyn.plot(
                x_test_loss_dyn, dyn_test_loss_trace, c="g", label="test loss"
            )
            if dyn_loss_trace[0] > 1 and dyn_loss_trace[-1] < 0.1:
                ax_loss_dyn.set_yscale("symlog")
            ax_loss_dyn.set_xlabel("epochs")
            ax_loss_dyn.set_ylabel("loss")
            add_grid(ax_loss_dyn)
            ax_loss_dyn.set_title(
                f"DYN {dyn_model_type} loss "
                f"({int(dyn_train_buffer.size/horizon)} episodes, lr={lr_dyn:.0e})"
            )
            ax_loss_dyn.legend()
            plt.savefig(results_dir / "dyn_loss.png")

        if policy_type != "tvlg":
            fig_loss_pol, ax_loss_pol = plt.subplots(**fig_kwargs)
            x_train_loss_pol = scaled_xaxis(pol_loss_trace, n_x_ticks=pol_epoch_counter)
            ax_loss_pol.plot(
                x_train_loss_pol, pol_loss_trace, c="k", label="train loss"
            )
            x_test_loss_pol = scaled_xaxis(
                pol_test_loss_trace, dist_x_ticks=n_epochs_between_eval_pol
            )
            ax_loss_pol.plot(
                x_test_loss_pol, pol_test_loss_trace, c="g", label="test loss"
            )
            if pol_loss_trace[0] > 1 and pol_loss_trace[-1] < 0.1:
                ax_loss_pol.set_yscale("symlog")
            ax_loss_pol.set_xlabel("epochs")
            ax_loss_pol.set_ylabel("loss")
            add_grid(ax_loss_pol)
            ax_loss_pol.set_title(
                f"POL {policy_type} loss "
                f"({int(pol_train_buffer.size/horizon)} local solutions, lr={lr_pol:.0e})"
            )
            ax_loss_pol.legend()
            plt.savefig(results_dir / "pol_loss.png")

        if show_plots:
            plt.show()

    #### modified initial state

    # original:
    # initial_state_distribution = MultivariateGaussian(
    #     torch.Tensor([torch.pi, 0.0]),
    #     torch.diag_embed(torch.Tensor([1e-2, 1e-6])),
    # )

    # new: (breaks local_policy!!)
    initial_state_distribution.covariance = torch.diag_embed(torch.Tensor([1, 1e-6]))

    n_mod_inits = 2
    mod_inits = initial_state_distribution.sample([n_mod_inits])

    ## plot local policies vs env
    xsl, usl, uvarsl = [], [], []
    for i in range(n_mod_inits):
        xs, us, xxs = environment.run(
            mod_inits[i, ...].repeat(n_i2c_vec, 1), local_vec_policy, horizon
        )
        uvars = []
        for t in range(horizon):
            u_dist = local_vec_policy.predict_dist(xs[t, ...], t=t)
            u_var = u_dist.covariance.diagonal(dim1=-2, dim2=-1)
            uvars.append(u_var)
        uvars = torch.stack(uvars)
        xsl.append(xs)
        usl.append(us)
        uvarsl.append(uvars)
        rollout_plot(xs, us, uvars=uvars, u_max=u_max, **fig_kwargs)
        plt.suptitle(
            f"{n_i2c_vec} local policies vs env ({n_mod_inits} modified init states[{i}])"
        )
        plt.savefig(results_dir / f"{n_i2c_vec}_tvlgs_vs_env_{i_iter}_modinit{i}.png")
    # xs = torch.cat(xsl, dim=-2)
    # us = torch.cat(usl, dim=-2)
    # uvars = torch.cat(uvarsl, dim=-2)
    # rollout_plot(xs, us, uvars=uvars, u_max=u_max, **fig_kwargs)
    # plt.suptitle(
    #     f"{n_i2c_vec} local policies vs env ({n_mod_inits} modified init states[{i}])"
    # )
    # plt.savefig(results_dir / f"{n_i2c_vec}_tvlgs_vs_env_{i_iter}_modinit.png")

    ### policy vs env
    xs, us, xxs = environment.run(mod_inits, global_policy, horizon)
    uvars = []
    for t in range(horizon):
        u_dist = global_policy.predict_dist(xs[t, ...], t=t)
        # u_dist = local_vec_policy.predict_dist(xs[t, ...], t=t)
        u_var = u_dist.covariance.diagonal(dim1=-2, dim2=-1)
        uvars.append(u_var)
    uvars = torch.stack(uvars)
    rollout_plot(xs, us, uvars=uvars, u_max=u_max, **fig_kwargs)
    plt.suptitle(f"{policy_type} policy vs env ({n_mod_inits} modified init states)")
    # plt.suptitle(f"{n_i2c_vec} local policies vs env (modified init state)")
    plt.savefig(results_dir / f"{policy_type}_vs_env_{i_iter}_modinit.png")

    ## mixture policy vs env
    xs = torch.zeros((horizon, n_mod_inits, dim_x))
    xvars = torch.zeros((horizon, n_mod_inits, dim_x))
    us = torch.zeros((horizon, n_mod_inits, dim_u))
    uvars = torch.zeros((horizon, n_mod_inits, dim_u))
    s_dist = MultivariateGaussian.from_deterministic(mod_inits)
    for t in range(horizon):
        xs[t, ...] = s_dist.mean
        xvars[t, ...] = s_dist.covariance.diagonal(dim1=-2, dim2=-1)
        a_dist, importance = mixture_tvlg(xs[t, ...], t=t)
        us[t, ...] = a_dist.mean
        uvars[t, ...] = a_dist.covariance.diagonal(dim1=-2, dim2=-1)
        # uvars[t, ...] = importance.log().sigmoid()
        xu = torch.cat((xs[t, ...], us[t, ...]), dim=-1)
        s_dist = global_dynamics.predict_dist(xu)
    # env.plot does not use env, it only plots
    rollout_plot(xs, us, xvars=xvars, uvars=uvars, u_max=u_max, **fig_kwargs)
    plt.suptitle(f"mixture policy vs env")
    plt.savefig(results_dir / f"mixture_tvlg_vs_env_{i_iter}.png")

    logger.strong_line()
    logger.info(f"Seed: {seed} - Took {time.time()-time_begin:.2f} seconds")
    logger.info(f"Logs in {results_dir}")
    logger.finish()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
