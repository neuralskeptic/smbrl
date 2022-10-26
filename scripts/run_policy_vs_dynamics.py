import json
import os

import gpytorch
import pandas as pd
import quanser_robots
import torch
import yaml
from experiment_launcher import run_experiment
from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor
from sklearn.model_selection import train_test_split
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from src.models.gp_models import ExactGPModel
from src.models.linear_bayesian_models import (
    NeuralLinearModel,
    SpectralNormalizedNeuralGaussianProcess,
)
from src.utils.conversion_utils import df2torch, np2torch, qube_rollout2df
from src.utils.replay_agent import replay_agent
from src.utils.seeds import fix_random_seed


def render_policy(
    sac_policy_dir: str = "models/2022_07_15__14_57_42",
    snngp_policy_dir: str = "logs/tmp/snngp_clone_SAC/0/2022_10_21__22_05_47",
    nlm_policy_dir: str = "logs/tmp/nlm_clone_SAC/0/2022_10_21__19_40_42",
    snngp_dynamics_dir_0: str = "logs/tmp/snngp_learn_dynamics/0/2022_10_22__03_49_31",
    snngp_dynamics_dir_1: str = "logs/tmp/snngp_learn_dynamics/0/2022_10_22__05_13_20",
    snngp_dynamics_dir_2: str = "logs/tmp/snngp_learn_dynamics/0/2022_10_22__06_37_18",
    snngp_dynamics_dir_3: str = "logs/tmp/snngp_learn_dynamics/0/2022_10_22__08_00_46",
    snngp_dynamics_dir_4: str = "logs/tmp/snngp_learn_dynamics/0/2022_10_22__09_24_45",
    snngp_dynamics_dir_5: str = "logs/tmp/snngp_learn_dynamics/0/2022_10_22__10_48_50",
    policy_alg: str = "snngp",  # of ['sac', 'snngp', 'nlm']
    dynamics_alg: str = "gym",  # of ['gym', 'snngp']
    use_cuda: bool = True,  # gp too slow on cpu
    n_runs: int = 10,
    # render: bool = True,
    render: bool = False,
    # plot: bool = False,
    plot: bool = True,
    show_plots: bool = False,
    seed: int = 0,
    results_dir: str = "",  ## IGNORED (only needed to run with with run_experiment)
):
    ####################################################################################################################
    #### SETUP

    # Fix seeds
    fix_random_seed(seed)

    # dirs & paths
    repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)
    sac_agent_path = os.path.join(repo_dir, sac_policy_dir, "agent_end.msh")
    snngp_agent_path = os.path.join(repo_dir, snngp_policy_dir, "agent_end.pth")
    nlm_agent_path = os.path.join(repo_dir, nlm_policy_dir, "agent_end.pth")
    snngp_dyn_paths = [
        os.path.join(repo_dir, snngp_dynamics_dir_0, "agent_end.pth"),
        os.path.join(repo_dir, snngp_dynamics_dir_1, "agent_end.pth"),
        os.path.join(repo_dir, snngp_dynamics_dir_2, "agent_end.pth"),
        os.path.join(repo_dir, snngp_dynamics_dir_3, "agent_end.pth"),
        os.path.join(repo_dir, snngp_dynamics_dir_4, "agent_end.pth"),
        os.path.join(repo_dir, snngp_dynamics_dir_5, "agent_end.pth"),
    ]

    # load configs
    def load_config(config_dir):
        try:
            with open(os.path.join(repo_dir, config_dir, "args.json")) as f:
                return json.load(f)
        except:
            with open(os.path.join(repo_dir, config_dir, "args.yaml")) as f:
                return yaml.load(f, Loader=yaml.Loader)

    sac_args = load_config(sac_policy_dir)
    snngp_args = load_config(snngp_policy_dir)
    nlm_args = load_config(nlm_policy_dir)
    snngp_dyn_args = [
        load_config(snngp_dynamics_dir_0),
        load_config(snngp_dynamics_dir_1),
        load_config(snngp_dynamics_dir_2),
        load_config(snngp_dynamics_dir_3),
        load_config(snngp_dynamics_dir_4),
        load_config(snngp_dynamics_dir_5),
    ]

    # dims
    s_dim = 6
    a_dim = 1

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    ####################################################################################################################
    #### EXPERIMENT SETUP

    # MDP
    mdp = Gym(sac_args["env_id"], horizon=sac_args["horizon"], gamma=sac_args["gamma"])
    mdp.seed(seed)

    # Agent
    if policy_alg == "sac":
        sac_agent = Agent.load(sac_agent_path)
        core = Core(sac_agent, mdp)

        def policy(state):
            action = core.agent.policy.compute_action_and_log_prob_t(
                state, compute_log_prob=False, deterministic=True
            )
            return action.cpu().reshape(-1).numpy()

    elif policy_alg == "nlm":
        agent = NeuralLinearModel(
            s_dim, a_dim, nlm_args["n_features"], nlm_args["lr"], device=device
        )
        state_dict = torch.load(nlm_agent_path)
        agent.load_state_dict(state_dict)

        def policy(state):
            state_torch = np2torch(state).reshape(1, -1).to(device)
            mu, _, _, _ = agent(state_torch)
            return mu.cpu().reshape(-1).numpy()

    elif policy_alg == "snngp":
        agent = SpectralNormalizedNeuralGaussianProcess(
            s_dim, a_dim, snngp_args["n_features"], snngp_args["lr"], device=device
        )
        state_dict = torch.load(snngp_agent_path)
        agent.load_state_dict(state_dict)

        def policy(state):
            state_torch = np2torch(state).reshape(1, -1).to(device)
            mu, _, _, _ = agent(state_torch)
            return mu.cpu().reshape(-1).numpy()

    else:
        raise Exception(f"Agent algorithm -{policy_alg}- unknown. Aborting")

    # Dynamics
    if dynamics_alg == "snngp":
        dyn_models = []
        for i in range(6):  # n dynamics models
            args = snngp_dyn_args[i]
            model_path = snngp_dyn_paths[i]
            model = SpectralNormalizedNeuralGaussianProcess(
                s_dim + a_dim,
                1,
                snngp_args["n_features"],
                snngp_args["lr"],
                device=device,
            )
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            dyn_models.append(model)

        def dynamics(state, action):
            ##### qube.step(a) ##### (self._state = th, al, thd, ald)
            # rwd, done = self._rwd(self._state, a)
            # self._state, act = self._ctrl_step(a)
            # obs = np.float32([np.cos(self._state[0]), np.sin(self._state[0]),
            #                   np.cos(self._state[1]), np.sin(self._state[1]),
            #                   self._state[2], self._state[3]])
            # return obs, rwd, done, {'s': self._state, 'a': act}
            ########################
            state_torch = np2torch(state).reshape([-1, s_dim]).to(device)
            action_torch = np2torch(action).reshape([-1, a_dim]).to(device)
            state_action = torch.cat((state_torch, action_torch), dim=1)
            next_state_torch = torch.empty(s_dim)
            for si in range(s_dim):
                mu, _, _, _ = dyn_models[si](state_action)
                next_state_torch[si] = mu + state_torch[0, si]
            next_state = next_state_torch.cpu().reshape(-1).numpy()
            next_state4 = state6to4(next_state)
            reward, absorbing = mdp.env.unwrapped._rwd(next_state4, action)
            # next_state, reward, absorbing, last
            return next_state, reward, False, False

    elif dynamics_alg == "gym":

        def dynamics(state, action):
            return mdp.step(action)

    else:
        raise Exception(f"Dynamics algorithm -{dynamics_alg}- unknown. Aborting")

    ####################################################################################################################
    #### RUNNING & PLOTTING

    if plot:
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

    for i in range(n_runs):
        ##### RUN AGENT for one episode #####
        dataset = list()
        episode_steps = 0
        state = mdp.reset(None).copy()

        last = False
        while not last:
            with torch.no_grad():
                action = policy(state)
                next_state, reward, absorbing, _ = dynamics(state, action)
            if render:
                mdp.render()
            episode_steps += 1
            if episode_steps >= mdp.info.horizon or absorbing:
                last = True
            sample = (state, action, reward, next_state, absorbing, last)
            dataset.append(sample)
            state = next_state.copy()

        mdp.stop()

        # MAX_STEPS = 100 + 80
        if plot:
            s, a, r, ss, absorb, last = parse_dataset(dataset)
            # axs[0].plot(r[:MAX_STEPS])
            # axs[1].plot(a[:MAX_STEPS])
            axs[0].plot(r)
            axs[1].plot(a)
    if plot:
        axs[0].set_title(f"run {policy_alg} policy against {dynamics_alg} dynamics")
        axs[0].grid(True)
        axs[1].grid(True)
        axs[1].set_xlabel("steps")
        axs[0].set_ylabel("reward")
        axs[1].set_ylabel("action")
        # plt.savefig(
        #     os.path.join(
        #         repo_dir, results_dir, f"run_policy_{alg}_ep{agent_epoch}.png"
        #     ),
        #     dpi=150,
        # )

    if show_plots:
        plt.show()


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(render_policy)
