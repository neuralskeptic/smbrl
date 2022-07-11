import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from experiment_launcher import run_experiment, save_args
from mushroom_rl.core import Core
from mushroom_rl.core.logger.logger import Logger
from mushroom_rl.environments import Gym
from mushroom_rl.utils.callbacks import PlotDataset
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

import wandb
from scripts.exploration.exp_sac_base import evaluate_policy
from scripts.off_policy.sac_networks import ActorNetwork, CriticNetworkQfunction
from src.algs.step_based.msg_pg import MSG_PolicyGradient
from src.utils.seeds import fix_random_seed

########################################################################################################################
from src.utils.set_evaluation import SetEval
from src.utils.set_logging import SetLogging


def experiment(
    alg: str = "msg",
    env_id: str = "Pendulum-v1",
    horizon: int = 200,
    gamma: float = 0.99,
    n_epochs: int = 100,
    n_steps: int = 1000,
    n_episodes_test: int = 10,
    initial_replay_size: int = 256,
    max_replay_size: int = 500000,
    batch_size: int = 64,
    warmup_transitions: int = 256,
    tau: float = 0.005,
    lr_alpha: float = 3e-4,
    n_features_actor: int = 64,
    n_features_critic: int = 64,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    n_critics_ensemble: int = 5,
    n_epochs_critic: int = 1,
    m_ensemble: int = 2,
    policy_grad_reduction: str = "min",
    mc_samples_gradient: int = 1,
    coupling: bool = False,
    beta_UB: float = 1.0,
    kl_upper_bound: float = 6.86,
    lambda_kl: float = 0.01,
    lambda_entropy: float = 0.01,
    entropy_lower_bound: float = -10.0,
    mvd_random_directions: int = -1,
    exploration_policy_class: str = "policy",
    ucb_coeff: float = 1.0,
    use_entropy: bool = False,
    alpha_q_regularizer: float = 0.1,
    beta_ucb: float = -4.0,
    preprocess_states: bool = False,
    use_cuda: bool = False,
    debug: bool = False,
    verbose: bool = False,
    log_wandb: bool = False,
    wandb_project: str = None,
    wandb_entity: str = None,
    seed: int = 0,
    results_dir: str = "./logs/tmp/exp_msg/pendulum/",
):

    ####################################################################################################################
    # SETUP
    # Results directory
    results_dir = os.path.join(results_dir, str(seed))
    os.makedirs(results_dir, exist_ok=True)
    # Save arguments
    save_args(results_dir, locals(), git_repo_path="./")

    # logger
    logger = Logger(
        config=locals(),
        log_name=seed,
        results_dir=results_dir,
        project=wandb_project,
        entity=wandb_entity,
        tags=["env_id", "entity", "exploration_policy_class"],
        log_console=verbose,
        log_wandb=log_wandb,
    )

    ####################################################################################################################
    # EXPERIMENT
    s = time.time()

    if use_cuda:
        torch.set_num_threads(1)

    print(f"Env id: {env_id}, Alg: {alg}, Seed: {seed}")

    # MDP
    mdp = Gym(env_id, horizon, gamma)

    # Fix seed
    fix_random_seed(seed, mdp)

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(
        network=ActorNetwork,
        n_features=n_features_actor,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
        use_cuda=use_cuda,
    )
    actor_sigma_params = dict(
        network=ActorNetwork,
        n_features=n_features_actor,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
        use_cuda=use_cuda,
    )

    actor_optimizer = {"class": optim.Adam, "params": {"lr": lr_actor}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(
        network=CriticNetworkQfunction,
        optimizer={"class": optim.Adam, "params": {"lr": lr_critic}},
        loss=F.mse_loss,
        n_features=n_features_critic,
        input_shape=critic_input_shape,
        output_shape=(1,),
        n_models=n_critics_ensemble,
        use_cuda=use_cuda,
    )

    # Agent
    mc_gradient_estimator = {"estimator": "reptrick", "n_samples": mc_samples_gradient}

    # Exploration policy options
    exploration_policy = {"name": "policy"}
    if exploration_policy_class == "policy_oac":
        exploration_policy = {
            "name": "policy_oac",
            "beta_UB": beta_UB,
            "kl_upper_bound": kl_upper_bound,
        }
    elif exploration_policy_class == "policy_troac":
        exploration_policy = {
            "name": "policy_troac",
            "beta_UB": beta_UB,
            "kl_upper_bound": kl_upper_bound,
        }
    elif exploration_policy_class == "policy_troac_reptrick":
        exploration_policy = {
            "name": "policy_troac_reptrick",
            "beta_UB": beta_UB,
            "kl_upper_bound": kl_upper_bound,
            "lambda_kl": lambda_kl,
            "lambda_entropy": lambda_entropy,
        }
    elif exploration_policy_class == "policy_random_bounds":
        exploration_policy = {"name": "policy_random_bounds"}

    agent = MSG_PolicyGradient(
        mdp.info,
        actor_mu_params,
        actor_sigma_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
        critic_fit_params=dict(n_epochs=n_epochs_critic),
        mc_gradient_estimator=mc_gradient_estimator,
        exploration_policy=exploration_policy,
        use_entropy=use_entropy,
        alpha_q_regularizer=alpha_q_regularizer,
        beta_ucb=beta_ucb,
    )

    # Save the agent before training
    agent.save(os.path.join(results_dir, "agent_begin.msh"), full_save=True)

    # Algorithm
    prepro = None
    if preprocess_states:
        prepro = StandardizationPreprocessor(mdp_info=mdp.info)

    plotter = None
    if debug:
        plotter = PlotDataset(mdp.info, obs_normalized=True)

    core = Core(
        agent,
        mdp,
        callback_step=plotter,
        preprocessors=[prepro] if prepro is not None else None,
    )

    # TRAIN
    J_l = []
    R_l = []

    # Evaluate before learning
    J_l, R_l = evaluate_policy(
        0,
        n_steps,
        n_epochs,
        agent,
        core,
        gamma,
        n_episodes_test,
        verbose,
        J_l,
        R_l,
        logger,
        seed,
    )

    # Fill up the replay memory
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in range(1, n_epochs + 1):
        with SetLogging(logger, [agent.exploration_policy]):
            # Learn and evaluate
            core.learn(
                n_steps=n_steps, n_steps_per_fit=1, quiet=not verbose, render=False
            )
        J_l, R_l = evaluate_policy(
            n,
            n_steps,
            n_epochs,
            agent,
            core,
            gamma,
            n_episodes_test,
            verbose,
            J_l,
            R_l,
            logger,
            seed,
        )

        # save the current logged data to keep an intermediate results if the training breaks
        logger.save_logs()

    # Save the agent after training
    agent.save(os.path.join(results_dir, "agent_end.msh"), full_save=True)

    logger.finish()

    print(f"Seed: {seed} - Took {time.time()-s:.2f} seconds")


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
