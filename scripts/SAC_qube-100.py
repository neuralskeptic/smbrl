import numpy as np
import quanser_robots
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from tqdm import trange


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def qube_rwd(self, x, a):
    th, al, thd, ald = x
    al_mod = al % (2 * np.pi) - np.pi
    done = not self.state_space.contains(x)

    factor = [0.96, 0.039, 0.001]  # [0.9, 0.05, 0.05]
    scales = [np.pi, 2.0, 5.0]

    err_dist = th
    err_rot = al_mod
    err_act = a[0]

    rotation_rew = (1 - np.abs(err_rot / scales[0])) ** 2
    distance_rew = (1 - np.abs(err_dist / scales[1])) ** 2
    action_rew = (1 - np.abs(err_act / scales[2])) ** 2

    # Reward should be roughly between [0, 1]
    rew = factor[0] * rotation_rew + factor[1] * distance_rew + factor[2] * action_rew
    return np.float32(np.clip(rew, 0, 1)), done


def experiment(alg, n_epochs, n_steps, n_steps_test, env_name="Qube-100-v0"):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: " + alg.__name__)

    # MDP
    horizon = 3000
    gamma = 0.99
    mdp = Gym(env_name, horizon, gamma)

    # mdp.env.reward_range = (0.0, 1.0)
    # mdp.env._rew = qube_rwd        # this does nothing, has to be changed in clients/quanser_robots/qube/base.py:_rwd()
    print(mdp.env.timing.dt_ctrl, mdp.env.reward_range)

    # Settings
    initial_replay_size = 3000  # 64
    max_replay_size = 500000
    batch_size = 256
    n_features = 64
    warmup_transitions = 3000
    tau = 0.005
    lr_alpha = 5e-5  # 3e-4

    use_cuda = torch.cuda.is_available()
    print(torch.__version__)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
        use_cuda=use_cuda,
    )
    actor_sigma_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=actor_input_shape,
        output_shape=mdp.info.action_space.shape,
        use_cuda=use_cuda,
    )

    print(mdp.info.action_space.low, mdp.info.action_space.high)

    actor_optimizer = {"class": optim.Adam, "params": {"lr": 3e-4}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(
        network=CriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
        use_cuda=use_cuda,
    )

    # Agent
    agent = alg(
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
        critic_fit_params=None,
    )

    print(agent.policy._delta_a, agent.policy._central_a)

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    s, *_ = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)

    logger.epoch_info(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        s, *_ = parse_dataset(dataset)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)

        # actions = parse_dataset(dataset)[1]
        # min_a = np.min(actions)
        # max_a = np.max(actions)
        max_r = np.max(parse_dataset(dataset)[2])

        logger.epoch_info(
            n + 1, J=J, R=R, entropy=E, max_r=max_r
        )  # , min_a=min_a, max_a=max_a)

    logger.info("Press a button to visualize pendulum")
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == "__main__":
    algs = [SAC]

    for alg in algs:
        # experiment(alg=alg, n_epochs=200, n_steps=2000, n_steps_test=2000, env_name='CartpoleSwingShort-v0')

        experiment(
            alg=alg,
            n_epochs=240,
            n_steps=2000,
            n_steps_test=2000,
            env_name="Qube-500-v0",
        )
