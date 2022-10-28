import numpy as np
import torch


def state4to6(state4):
    state4 = state4.reshape(4, -1)
    n = state4.shape[1]
    if n == 1:
        state6 = np.empty((6,))
    else:
        state6 = np.empty((6, n))
    state6[0] = np.cos(state4[0])
    state6[1] = np.sin(state4[0])
    state6[2] = np.cos(state4[1])
    state6[3] = np.sin(state4[1])
    state6[4] = state4[2]
    state6[5] = state4[3]
    if n == 1:
        return state6
    else:
        return state6.reshape(-1, 6)


def state6to4(state6):
    state6 = state6.reshape(6, -1)
    n = state6.shape[1]
    if n == 1:
        state4 = np.empty((4,))
    else:
        state4 = np.empty((4, n))
    state4[0] = np.arctan2(state6[1], state6[0])
    state4[1] = np.arctan2(state6[3], state6[2])
    state4[2] = state6[4]
    state4[3] = state6[5]
    if n == 1:
        return state4
    else:
        return state4.reshape(-1, 4)


def rollout(mdp, policy, n_episodes=1):
    """


    Parameters
    ----------
    init : lambda -> state4
        get initial state
    policy : lambda state4 -> action

    dynamics : lambda state4, action -> next_state4

    n_episodes : int, optional
        The default is 1.

    Returns
    -------
    dataset - list of (state4, action, reward, next_state4, absorbing, last)
        All sampled transitions in chronological order, where only the last
        transition of every episode has last=True.

    """
    dataset = list()
    for i in range(n_episodes):
        episode_steps = 0
        last = False
        state4 = state6to4(mdp.reset(None).copy())
        while last is False:
            action = policy(state4)
            if action is None:
                breakpoint()
            _, reward, absorbing, ssa_dict = mdp.step(action)
            next_state4 = ssa_dict["s"]  # angle, not cos/sin
            episode_steps += 1
            if episode_steps >= mdp.info.horizon:
                last = True
            sample = (state4, action, reward, next_state4, absorbing, last)
            dataset.append(sample)
            state4 = next_state4.copy()
    return dataset
