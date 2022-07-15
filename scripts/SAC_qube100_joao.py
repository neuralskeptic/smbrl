import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from experiment_launcher import run_experiment
from experiment_launcher.utils import save_args
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core
from mushroom_rl.core.logger.logger import Logger
from mushroom_rl.environments import Gym
from mushroom_rl.utils.callbacks import PlotDataset
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

from src.models.sac_networks import ActorNetwork, CriticNetworkQfunction
from src.utils.evaluate_policy import evaluate_policy
from src.utils.seeds import fix_random_seed
from src.utils.set_logging import SetLogging
from src.utils.time_utils import timestamp


def experiment(
    alg: str = "SAC",
    env_id: str = "Qube-100-v0",
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
    n_critics_ensemble: int = 2,
    n_epochs_critic: int = 1,
    m_ensemble: int = 2,
    preprocess_states: bool = False,
    use_cuda: bool = False,
    plotting: bool = False,  # crashes ipython kernel???
    verbose: bool = False,
    model_save_frequency: bool = 1,  # every x epochs
    log_wandb: bool = True,
    wandb_project: str = "smbrl",
    wandb_entity: str = "showmezeplozz",
    wandb_group: str = "SAC",
    wandb_job_type: str = "train",
    seed: int = 0,
    results_dir: str = "./logs/tmp/",
    debug: bool = False,
):
    # every forked process needs to register gym envs
    import quanser_robots

    # but remove the module from locals(), since it cannot be serialized
    locals_without_quanser = locals()
    del locals_without_quanser["quanser_robots"]

    ####################################################################################################################
    # SETUP
    s = time.time()

    if debug:
        # disable wandb logging and redirect normal logging to ./debug directory
        print("@@@@@@@@@@@@@@@@@ DEBUG: LOGGING DISABLED @@@@@@@@@@@@@@@@@")
        os.environ["WANDB_MODE"] = "disabled"
        results_dir = os.path.join("debug", results_dir)

    # Results directory
    results_dir = os.path.join(results_dir, wandb_group, env_id, str(seed), timestamp())
    os.makedirs(results_dir, exist_ok=True)
    # Save arguments
    save_args(results_dir, locals_without_quanser, git_repo_path="./")

    # logger
    logger = Logger(
        config=locals_without_quanser,
        log_name=seed,
        results_dir=results_dir,
        project=wandb_project,
        entity=wandb_entity,
        wandb_kwargs={"group": wandb_group, "job_type": wandb_job_type},
        tags=["env_id", "entity"],
        log_console=verbose,
        log_wandb=log_wandb,
    )

    ####################################################################################################################
    # EXPERIMENT

    if use_cuda:
        torch.set_num_threads(1)
    # print(torch.device)

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
    # print(mdp.info.action_space.low, mdp.info.action_space.high)

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
    agent = SAC(
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
    )

    # Save the agent before training
    agent.save(os.path.join(results_dir, "agent_begin.msh"), full_save=False)

    # Algorithm
    prepro = None
    if preprocess_states:
        prepro = StandardizationPreprocessor(mdp_info=mdp.info)

    plotter = None
    if plotting:
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
        with SetLogging(
            logger, [agent.policy]
        ):  # TODO check if okay (was exploration_policy)
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
        if n % model_save_frequency == 0:
            # Save the agent
            agent.save(os.path.join(results_dir, f"agent_{n}.msh"), full_save=False)

    # Save the agent after training
    agent.save(os.path.join(results_dir, "agent_end.msh"), full_save=False)

    # policy execution reward plot
    if log_wandb:
        import pandas as pd
        from mushroom_rl.utils.dataset import parse_dataset

        import wandb
        from src.utils.replay_agent import replay_agent

        rollouts = replay_agent(agent, core, 1, verbose=False, render=False)
        state, action, reward, nstate, absorb, last = parse_dataset(rollouts)
        MAX_STEPS = 80 + 100
        data = {
            "steps": torch.arange(MAX_STEPS),
            "reward": reward[:MAX_STEPS],
            "action": action[:MAX_STEPS, 0],
        }
        wandb.log({"evaluation": wandb.Table(dataframe=pd.DataFrame(data))})
        # visualized in custom chart taking data from run.summary evaluation table

    logger.finish()

    print(f"Seed: {seed} - Took {time.time()-s:.2f} seconds")
    print(f"Logs in {results_dir}")


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
