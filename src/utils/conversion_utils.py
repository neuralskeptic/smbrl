import numpy as np
import pandas as pd
import torch
from mushroom_rl.utils.dataset import parse_dataset


def np2torch(x):
    return torch.from_numpy(x).float()


def df2torch(df):
    return torch.tensor(df.values)


def vec(x):
    return x.T.reshape(-1)


def autograd_tensor(x):
    """
    Same as torch.Tensor(x, requires_grad=True), but does not cause warnings.
    detach first and then clonw removes the clone from the computation graph
    """
    return x.detach().clone().requires_grad_(True)


def qube_rollout2df(data):
    s, a, r, ss, absorb, last = parse_dataset(data)
    N = len(a)
    df = pd.DataFrame()
    df[["s0", "s1", "s2", "s3", "s4", "s5"]] = s.reshape(N, -1)
    df[["a"]] = a.reshape(N, -1)
    df[["r"]] = r.reshape(N, -1)
    df[["ss0", "ss1", "ss2", "ss3", "ss4", "ss5"]] = ss.reshape(N, -1)
    df[["absorb"]] = absorb.reshape(N, -1)
    df[["last"]] = last.reshape(N, -1)
    return df


def dataset2df_4(data):
    s, a, r, ss, absorb, last = parse_dataset(data)
    df = pd.DataFrame()
    df["theta"] = s[:, 0]
    df["alpha"] = s[:, 1]
    df["theta_dot"] = s[:, 2]
    df["alpha_dot"] = s[:, 3]
    df["a"] = a.reshape(-1)
    df["r"] = r.reshape(-1)
    df["next_theta"] = ss[:, 0]
    df["next_alpha"] = ss[:, 1]
    df["next_theta_dot"] = ss[:, 2]
    df["next_alpha_dot"] = ss[:, 3]
    df["absorb"] = absorb.reshape(-1)
    df["last"] = last.reshape(-1)
    # last_idxs = np.argwhere(last == 1).ravel()
    # n_episodes = len(last_idxs)
    # sub_dataset = dataset[:last_idxs[n_episodes - 1] + 1, :]
    n = len(a)
    traj_ids = np.empty(n)
    cur_id = 0
    for i in range(n):
        traj_ids[i] = cur_id
        if last[i]:
            cur_id += 1
    df["traj_id"] = traj_ids
    return df


def map_cpu(iterable):
    return map(lambda x: x.to("cpu"), iterable)
