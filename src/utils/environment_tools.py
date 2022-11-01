import numpy as np
import torch
from tqdm import tqdm


def state4to6(state4):
    transposed = False
    if len(state4.shape) == 1:  # (4,) -> (4, 1)
        state4 = state4.reshape(4, 1)
    if state4.shape[0] != 4:  # (n, 4) -> (4, n)
        state4 = state4.transpose()
        transposed = True
    n = state4.shape[1]
    state6 = np.empty((6, n))
    state6[0, :] = np.cos(state4[0, :])
    state6[1, :] = np.sin(state4[0, :])
    state6[2, :] = np.cos(state4[1, :])
    state6[3, :] = np.sin(state4[1, :])
    state6[4, :] = state4[2, :]
    state6[5, :] = state4[3, :]
    if n == 1:
        return state6.reshape((6,))  # (6,1) -> (6,)
    if transposed:  # input (n, 4)
        return state6.transpose()  # (6, n) -> (n, 6)
    return state6  # input (4, n), output (6, n)


def state6to4(state6):
    transposed = False
    if len(state6.shape) == 1:  # (6,) -> (6, 1)
        state6 = state6.reshape(6, 1)
    if state6.shape[0] != 6:  # (n, 6) -> (6, n)
        state6 = state6.transpose()
        transposed = True
    n = state6.shape[1]
    state4 = np.empty((4, n))
    state4[0, :] = np.arctan2(state6[1, :], state6[0, :])
    state4[1, :] = np.arctan2(state6[3, :], state6[2, :])
    state4[2, :] = state6[4, :]
    state4[3, :] = state6[5, :]
    if n == 1:
        return state4.reshape((4,))  # (4,1) -> (4,)
    if transposed:  # input (n, 4)
        return state4.transpose()  # (4, n) -> (n, 4)
    return state4  # input (6, n), output (4, n)


def rollout(mdp, policy, n_episodes=1, show_progress=False):
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
    for i in tqdm(range(n_episodes), disable=not show_progress):
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
