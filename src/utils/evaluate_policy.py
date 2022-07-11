import numpy as np
from mushroom_rl.utils.dataset import compute_J

from src.utils.set_evaluation import SetEval


def evaluate_policy(
    itr,
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
    seed=0,
):
    with SetEval(agent):
        data = core.evaluate(
            n_episodes=n_episodes_test, quiet=not verbose, render=False
        )
    J = compute_J(data, gamma)
    R = compute_J(data)

    n_samples = itr * n_steps
    J_l.append((n_samples, np.mean(J)))
    R_l.append((n_samples, np.mean(R)))

    print(
        f"Seed: {seed:4d} - Epoch: {itr}/{n_epochs} - samples: {n_samples:6d} -"
        f" J: {np.mean(J):.4f} - R: {np.mean(R):.4f}"
    )

    # log R and J to file and WandB if available
    logger.log_data(step=n_samples, R=np.mean(R), J=np.mean(J))

    return J_l, R_l
