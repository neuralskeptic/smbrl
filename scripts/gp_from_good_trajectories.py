import os
import time

import GPy
import gpytorch
import matplotlib.pyplot as plt
import pandas as pd
import torch
from experiment_launcher import run_experiment
from experiment_launcher.utils import save_args
from sklearn.model_selection import train_test_split

from src.utils.conversion_utils import df2torch, map_cpu
from src.utils.plotting_utils import plot_gp
from src.utils.seeds import fix_random_seed
from src.utils.time_utils import timestamp


def experiment(
    alg: str = "gp",
    dataset_file: str = "models/2022_07_15__14_57_42/SAC_on_Qube-100-v0_100trajs.pkl.gz",
    n_trajectories: int = 15,  # 80% train, 20% test
    n_epochs: int = 10,
    lr: float = 1e-1,
    use_cuda: bool = False,
    # verbose: bool = False,
    # model_save_frequency: bool = 5,  # every x epochs
    # log_wandb: bool = True,
    # wandb_project: str = "smbrl",
    # wandb_entity: str = "showmezeplozz",
    wandb_group: str = "gp_clone_SAC",
    # wandb_job_type: str = "train",
    seed: int = 0,
    results_dir: str = "logs/tmp/",
    debug: bool = True,
):
    ####################################################################################################################
    # SETUP
    time_begin = time.time()

    if debug:
        # disable wandb logging and redirect normal logging to ./debug directory
        print("@@@@@@@@@@@@@@@@@ DEBUG: LOGGING DISABLED @@@@@@@@@@@@@@@@@")
        os.environ["WANDB_MODE"] = "disabled"
        results_dir = os.path.join("debug", results_dir)

    # Fix seeds
    fix_random_seed(seed)

    # Results directory
    repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)
    results_dir = os.path.join(
        repo_dir, results_dir, wandb_group, str(seed), timestamp()
    )
    os.makedirs(results_dir, exist_ok=True)

    # Save arguments
    save_args(results_dir, locals(), git_repo_path="./")

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    ####################################################################################################################
    # EXPERIMENT

    print(f"Alg: {alg}, Seed: {seed}, Dataset: {dataset_file}")

    # df: [s0-s5, a, r, ss0-ss5, absorb, last]
    df = pd.read_pickle(os.path.join(repo_dir, dataset_file))
    df = df[df["traj_id"] < n_trajectories]  # take only n_trajectories
    df = df.astype("float32")  # otherwise spectral_norm complains
    traj_dfs = [
        traj.reset_index(drop=True) for (traj_id, traj) in df.groupby("traj_id")
    ]
    train_traj_dfs, test_traj_dfs = train_test_split(traj_dfs, test_size=0.2)
    x_cols = ["s0", "s1", "s2", "s3", "s4", "s5"]
    y_cols = ["a"]
    train_df = pd.concat(train_traj_dfs)
    test_df = pd.concat(test_traj_dfs)
    train_x_df, train_y_df = train_df[x_cols], train_df[y_cols]
    test_x_df, test_y_df = test_df[x_cols], test_df[y_cols]
    x_train, y_train = df2torch(train_x_df), df2torch(train_y_df)
    x_test, y_test = df2torch(test_x_df), df2torch(test_y_df)
    # reshape labels, because mean_prior implicitly flattens and it crashes
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    # # whiten data here for simplicity
    # x_mean, x_std = x_train.mean(), x_train.std()
    # y_mean, y_std = y_train.mean(), y_train.std()
    # x_train = (x_train - x_mean) / x_std
    # x_test = (x_test - x_mean) / x_std
    # y_train = (y_train) / y_std
    # y_test = (y_test) / y_std

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean(prior=None)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(lengthscale_prior=None),
                outputscale_prior=None,
            )

        def forward(self, x):
            # breakpoint()
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=None)
    model = ExactGPModel(x_train, y_train, likelihood)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # margll only works for GaussianLikelihood and ExactGP
    margll_loss = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    trace = []
    for i in range(n_epochs):
        # only full batches (for GP training)
        model.train()
        likelihood.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = -margll_loss(pred, y_train).sum()
        loss.backward()
        opt.step()

        # logging
        if i % 1 == 0:
            model.eval()  # needed for logprob predictions
            likelihood.eval()
            with torch.no_grad():
                trace.append(loss.detach().item())
                cur_ls = model.covar_module.base_kernel.lengthscale.item()
                cur_noise = model.likelihood.noise.item()
                # logprob fails with more than ~5 trajs (cholesky: cov not PSD)
                # cur_train_logprob = likelihood(model(x_train)).log_prob(y_train)
                # cur_test_logprob = likelihood(model(x_test)).log_prob(y_test)
                print(
                    f"Iter {i+1}/{n_epochs}, Loss: {loss.item():.3f}, "
                    f"ls: {cur_ls:.3f}, noise: {cur_noise:.3f}, "
                    # f"train logp: {cur_train_logprob:.2f}, "
                    # f"test logp: {cur_test_logprob:.2f}, "
                )

    ####################################################################################################################
    # EVALUATION
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        ## plot statistics over trajectories
        pred_post = likelihood(model(x_test.to(device)))
        test_df["pred"] = pred_post.mean.to("cpu").reshape(-1)
        # test_df['pred_var'] = pred_post.variance.to('cpu').reshape(-1)
        mean_traj = test_df.groupby(level=0).mean()
        std_traj = test_df.groupby(level=0).std()
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        MAX_TIME = 100
        x_time = torch.tensor(range(0, len(mean_traj)))[:MAX_TIME]
        y_data_mean = df2torch(mean_traj["a"])[:MAX_TIME]
        y_data_std = df2torch(std_traj["a"])[:MAX_TIME]
        y_pred_mu_mean = df2torch(mean_traj["pred"])[:MAX_TIME]
        y_pred_mu_std = df2torch(std_traj["pred"])[:MAX_TIME]
        # y_pred_var_mean = df2torch(mean_traj['pred_var'])[:MAX_TIME]
        # y_pred_var_sqrtmean = torch.sqrt(y_pred_var_mean)
        plot_gp(
            ax,
            x_time,
            y_data_mean,
            y_data_std,
            color="b",
            label="test data mean & std trajs",
        )
        # plot_gp(ax, x_time, y_pred_mu_mean, y_pred_var_sqrtmean, color='c', label="pred mean(mu) & sqrt(mean(sigma))")
        plot_gp(
            ax,
            x_time,
            y_pred_mu_mean,
            y_pred_mu_std,
            color="r",
            alpha=0.2,
            label="test pred mean(mu) & std(mu)",
        )
        ax.set_xlabel("time")
        ax.set_ylabel("a")
        ax.set_title(
            f"nlm ({len(train_traj_dfs)}/{len(test_traj_dfs)} episodes, {n_epochs} epochs)"
        )
        ax.legend()
        plt.savefig(
            os.path.join(results_dir, "mean-std_traj_plots__test_data_pred.png"),
            dpi=150,
        )

    with torch.no_grad():
        ## print train and test MAE, MSE, RMSE
        def compute_MAE_MSE_RMSE(pred, test, prefix=""):
            MAE = torch.abs(pred - test).mean()
            MSE = torch.pow(pred - test, 2).mean()
            RMSE = torch.sqrt(MSE)
            return (
                MAE,
                MSE,
                RMSE,
            )

        with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
            post_pred = model(x_train)
            MAE, MSE, RMSE = compute_MAE_MSE_RMSE(post_pred.mean, y_train)
            logstring = f"Train: MAE={MAE.item():.2f}, MSE={MSE.item():.2f}, RMSE={RMSE.item():.2f}"
            print(logstring)
            f.write(logstring + "\n")
            post_pred = model(x_test)
            MAE, MSE, RMSE = compute_MAE_MSE_RMSE(post_pred.mean, y_test)
            logstring = f"Test: MAE={MAE.item():.2f}, MSE={MSE.item():.2f}, RMSE={RMSE.item():.2f}"
            print(logstring)
            f.write(logstring + "\n")

    # # plot train and test data and prediction
    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(x_train, y_train, c="grey", marker="x", s=5, label="train")
    # ax.scatter(x_test, y_test, c="k", s=5, label="test")
    # # mu_train, sigma_train, _, _ = model(x_train)
    # # plt.errorbar(
    # #     x_train.reshape(-1),
    # #     mu_train.reshape(-1),
    # #     torch.diag(sigma_train),
    # #     fmt="none",
    # #     color="r",
    # #     label="pred",
    # # )
    # mu_test, sigma_test, _, _ = model(x_test)
    # plt.errorbar(
    #     x_test.reshape(-1),
    #     mu_test.reshape(-1),
    #     torch.diag(sigma_test),
    #     fmt="none",
    #     color="r",
    #     label="pred",
    # )
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.legend()
    # ax.set_title(f"snngp (N={n_datapoints}, {n_epochs} epochs)")

    # plot training loss
    with torch.no_grad():
        fig_trace, ax_trace = plt.subplots()
        ax_trace.plot(trace, c="k")
        # ax_trace.plot(trace, '.', c='k') # dots for every data point
        ax_trace.set_xlabel("epochs")
        ax_trace.set_title("loss")
        plt.savefig(os.path.join(results_dir, "loss.png"), dpi=150)

    # # plot features on test dataset (sorted for plotting)
    # x_test_sorted, _ = x_test.sort(dim=0)
    # f = model.features(x_test_sorted).detach()
    # fig_features, ax_features = plt.subplots(figsize=(12, 9))
    # ax_features.plot(x_test_sorted, f[:, ::10], alpha=0.25)
    # ax_features.set_title("features on test dataset")

    plt.show()

    print(f"Seed: {seed} - Took {time.time()-time_begin:.2f} seconds")
    print(f"Logs in {results_dir}")


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
