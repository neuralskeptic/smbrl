import torch


def plot_gp(axis, x, mu, var, label=""):
    axis.plot(x, mu, "b-", label=label)
    for n_std in range(1, 3):
        std = n_std * torch.sqrt(var.squeeze())
        mu = mu.squeeze()
        upper, lower = mu + std, mu - std
        axis.fill_between(
            x.squeeze(),
            upper.squeeze(),
            lower.squeeze(),
            where=upper > lower,
            color="b",
            alpha=0.3,
        )
