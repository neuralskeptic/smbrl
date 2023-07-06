import json
import os

import numpy as np
import torch
import yaml
from experiment_launcher import run_experiment
from matplotlib import pyplot as plt
from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import parse_dataset
from tqdm import tqdm

from src.models.dnns import DNN3
from src.models.linear_bayesian_models import (
    NeuralLinearModelResNet,
    SpectralNormalizedNeuralGaussianProcess,
)
from src.utils.conversion_utils import np2torch
from src.utils.environment_tools import state4to6, state6to4
from src.utils.seeds import fix_random_seed
from src.utils.time_utils import timestamp


def render_policy(
    sac_policy_dir: str = "models/2022_07_15__14_57_42",
    mlp_policy_dir: str = "models/mlp_clone_SAC/2022_11_17__00_53_25",
    nlm_policy_dir: str = "models/nlm_clone_SAC/2022_11_17__00_17_24",
    # snngp_policy_dir: str = "models/snngp_clone_SAC/2022_11_17__14_56_23",
    # snngp_policy_dir: str = "debug/logs/tmp/snngp_clone_SAC/0/2023_06_23__15_23_23",
    snngp_policy_dir: str = "debug/logs/tmp/snngp_clone_SAC/0/2023_06_23__16_38_26",
    mlp_dynamics_dir: str = "debug/logs/tmp/mlp_learn_dynamics/0/2022_11_10__05_31_35",  # no y whitening
    # mlp_dynamics_dir: str = "debug/logs/tmp/mlp_learn_dynamics/0/2022_11_10__05_30_01",  # y whitening
    nlm_dynamics_dir: str = "debug/logs/tmp/nlm_learn_dynamics/0/2022_11_15__20_00_34",  # assump 1 (exc)
    # nlm_dynamics_dir: str = "debug/logs/tmp/nlm_learn_dynamics/0/2022_11_15__19_56_40",  # assump 2 (ok)
    # nlm_dynamics_dir: str = "debug/logs/tmp/nlm_learn_dynamics/0/2022_11_15__19_58_51",  # assump 3 (ok)
    # snngp_dynamics_dir: str = "debug/logs/tmp/snngp_learn_dynamics/0/2022_11_10__04_23_45",
    snngp_dynamics_dir: str = "debug/logs/tmp/snngp_learn_dynamics/0/2023_06_22__18_48_15",
    policy_alg: str = "snngp",  # of ['sac', 'mlp', 'nlm', 'snngp']
    dynamics_alg: str = "gym",  # of ['gym', 'mlp', 'nlm', 'snngp']
    use_cuda: bool = True,  # gp too slow on cpu
    n_runs: int = 10,
    # render: bool = True,
    render: bool = False,
    # plot: bool = False,
    plot: bool = True,
    show_plots: bool = False,
    seed: int = 0,
    results_dir: str = "logs/tmp/",
):
    ####################################################################################################################
    #### SETUP

    # Fix seeds
    fix_random_seed(seed)

    # Results directory
    repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)
    results_dir = os.path.join(repo_dir, results_dir, str(seed), timestamp())
    os.makedirs(results_dir, exist_ok=True)

    # dirs & paths
    sac_agent_path = os.path.join(repo_dir, sac_policy_dir, "agent_end.msh")
    mlp_agent_path = os.path.join(repo_dir, mlp_policy_dir, "agent_end.pth")
    nlm_agent_path = os.path.join(repo_dir, nlm_policy_dir, "agent_end.pth")
    snngp_agent_path = os.path.join(repo_dir, snngp_policy_dir, "agent_end.pth")
    mlp_dynamics_path = os.path.join(repo_dir, mlp_dynamics_dir, "agent_end.pth")
    nlm_dynamics_path = os.path.join(repo_dir, nlm_dynamics_dir, "agent_end.pth")
    snngp_dynamics_path = os.path.join(repo_dir, snngp_dynamics_dir, "agent_end.pth")

    # load configs
    def load_config(config_dir):
        try:
            with open(os.path.join(repo_dir, config_dir, "args.json")) as f:
                return json.load(f)
        except:
            with open(os.path.join(repo_dir, config_dir, "args.yaml")) as f:
                return yaml.load(f, Loader=yaml.Loader)

    sac_args = load_config(sac_policy_dir)
    mlp_args = load_config(mlp_policy_dir)
    nlm_args = load_config(nlm_policy_dir)
    snngp_args = load_config(snngp_policy_dir)
    mlp_dyn_args = load_config(mlp_dynamics_dir)
    nlm_dyn_args = load_config(nlm_dynamics_dir)
    snngp_dyn_args = load_config(snngp_dynamics_dir)

    # dims
    s6_dim = 6
    s4_dim = 4
    a_dim = 1

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    ####################################################################################################################
    #### EXPERIMENT SETUP

    # MDP
    import quanser_robots

    mdp = Gym(sac_args["env_id"], horizon=sac_args["horizon"], gamma=sac_args["gamma"])

    # Agent
    if policy_alg == "sac":
        sac_agent = Agent.load(sac_agent_path)
        core = Core(sac_agent, mdp)

        def policy(state):
            action = core.agent.policy.compute_action_and_log_prob_t(
                state, compute_log_prob=False, deterministic=True
            )
            return action.cpu().reshape(-1).numpy()

    elif policy_alg == "mlp":
        agent = DNN3(s6_dim, a_dim, mlp_args["n_features"])
        agent.to(device)
        state_dict = torch.load(mlp_agent_path)
        agent.load_state_dict(state_dict)

        def policy(state):
            state_torch = np2torch(state).reshape(1, -1).to(device)
            action_torch = agent(state_torch)
            return action_torch.cpu().reshape(-1).numpy()

    elif policy_alg == "nlm":
        agent = NeuralLinearModelResNet(s6_dim, a_dim, nlm_args["n_features"])
        agent.to(device)
        state_dict = torch.load(nlm_agent_path)
        agent.load_state_dict(state_dict)

        def policy(state):
            state_torch = np2torch(state).reshape(1, -1).to(device)
            mu, covs = agent(state_torch)
            return mu.cpu().reshape(-1).numpy()

    elif policy_alg == "snngp":
        agent = SpectralNormalizedNeuralGaussianProcess(
            s6_dim,
            snngp_args["layer_spec"],
            snngp_args["n_rff_features"],
            a_dim,
        )
        agent.to(device)
        state_dict = torch.load(snngp_agent_path)
        agent.load_state_dict(state_dict)

        def policy(state):
            state_torch = np2torch(state).reshape(1, -1).to(device)
            mu, covs = agent(state_torch)
            return mu.cpu().reshape(-1).numpy()

    else:
        raise Exception(f"Agent algorithm -{policy_alg}- unknown. Aborting")

    # Dynamics
    ##### qube.step(a) ##### (self._state = th, al, thd, ald)
    # rwd, done = self._rwd(self._state, a)
    # self._state, act = self._ctrl_step(a)
    # obs = np.float32([np.cos(self._state[0]), np.sin(self._state[0]),
    #                   np.cos(self._state[1]), np.sin(self._state[1]),
    #                   self._state[2], self._state[3]])
    # return obs, rwd, done, {'s': self._state, 'a': act}
    ########################
    if dynamics_alg == "mlp":
        model = DNN3(
            d_in=s6_dim + a_dim,
            d_out=s4_dim,
            d_hidden=mlp_dyn_args["n_features"],
        )
        model.to(device)
        state_dict = torch.load(mlp_dynamics_path)
        model.load_state_dict(state_dict)

        def dynamics(state4, action):
            _x = np2torch(np.hstack([state4to6(state4), action]))
            ss_delta_pred = model(_x.to(device)).cpu()
            next_state4 = state4 + ss_delta_pred.numpy()
            reward, absorbing = mdp.env.unwrapped._rwd(next_state4, action)
            # next_state, reward, absorbing, last
            return next_state4, reward, False, False

    elif dynamics_alg == "nlm":
        model = NeuralLinearModelResNet(
            dim_x=s6_dim + a_dim,
            dim_y=s4_dim,
            dim_features=nlm_dyn_args["n_features"],
        )
        model.to(device)
        state_dict = torch.load(nlm_dynamics_path)
        model.load_state_dict(state_dict)

        def dynamics(state4, action):
            _x = np2torch(np.hstack([state4to6(state4), action]))
            ss_delta_pred, covs = model(_x.to(device))
            next_state4 = state4 + ss_delta_pred.cpu().numpy()
            reward, absorbing = mdp.env.unwrapped._rwd(next_state4, action)
            # next_state, reward, absorbing, last
            return next_state4, reward, False, False

    elif dynamics_alg == "snngp":
        model = SpectralNormalizedNeuralGaussianProcess(
            dim_x=s6_dim + a_dim,
            # layer_spec=[*(256,) * 2],
            layer_spec=snngp_dyn_args["layer_spec"],
            dim_features=snngp_dyn_args["n_features"],
            dim_y=s4_dim,
        )
        model.to(device)
        state_dict = torch.load(snngp_dynamics_path)
        model.load_state_dict(state_dict)

        def dynamics(state4, action):
            _x = np2torch(np.hstack([state4to6(state4), action]))
            ss_delta_pred, covs = model(_x.to(device))
            next_state4 = state4 + ss_delta_pred.cpu().numpy()
            reward, absorbing = mdp.env.unwrapped._rwd(next_state4, action)
            # next_state, reward, absorbing, last
            return next_state4, reward, False, False

    elif dynamics_alg == "gym":

        def dynamics(state, action):
            return mdp.step(action)

    else:
        raise Exception(f"Dynamics algorithm -{dynamics_alg}- unknown. Aborting")

    ####################################################################################################################
    #### RUNNING

    datasets = []
    for i in tqdm(range(n_runs)):
        ##### RUN AGENT for one episode #####
        dataset = list()
        episode_steps = 0
        mdp.seed(seed + i)
        state6 = mdp.reset().copy()
        state4 = state6to4(state6)

        last = False
        while not last:
            with torch.no_grad():
                state6 = state4to6(state4)
                action = policy(state6)
                if dynamics_alg == "gym":
                    _, reward, absorbing, ssa_dict = mdp.step(action)
                    next_state4 = ssa_dict["s"]
                else:
                    next_state4, reward, absorbing, _ = dynamics(state4, action)
            if render:
                if dynamics_alg != "gym":
                    mdp.env.unwrapped._state = next_state4  # move gym
                mdp.render()
            episode_steps += 1
            if episode_steps >= mdp.info.horizon or absorbing:
                last = True
            sample = (state4, action, reward, next_state4, absorbing, last)
            dataset.append(sample)
            state4 = next_state4.copy()

        datasets.append(dataset)
        mdp.stop()

    ####################################################################################################################
    #### PLOTTING

    if plot:
        fig, axs = plt.subplots(5, 2, figsize=(10, 7), dpi=150)
        for dataset in datasets:
            s, a, r, ss, absorb, last = parse_dataset(dataset)
            x_time = torch.tensor(range(0, len(s)))
            # plot rewards
            axs[0, 0].plot(x_time, r)
            # plot actions
            axs[0, 1].plot(x_time, a, color="b")
            for yi in range(s4_dim):
                # delta next state prediction
                axs[yi + 1, 0].plot(x_time, (ss - s)[:, yi])
                # next state prediction
                axs[yi + 1, 1].plot(x_time, ss[:, yi])
        axs[0, 0].set_ylabel("reward")
        axs[0, 1].set_ylabel("action")
        for yi in range(s4_dim):
            axs[yi + 1, 0].set_ylabel(f"delta ss[{yi}]")
            axs[yi + 1, 1].set_ylabel(f"ss[{yi}]")
        axs[-1, 0].set_xlabel("steps")
        axs[-1, 1].set_xlabel("steps")
        title = f"run {n_runs}x {policy_alg} policy against {dynamics_alg} dynamics"
        fig.suptitle(title)
        plt.savefig(os.path.join(results_dir, f"{title}.png"), dpi=150)

    if show_plots:
        plt.show()

    print(f"Logs in {results_dir}")


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(render_policy)
