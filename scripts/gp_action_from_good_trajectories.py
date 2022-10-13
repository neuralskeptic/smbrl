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

from src.models.gp_models import ExactGPModel
from src.utils.conversion_utils import df2torch, map_cpu
from src.utils.plotting_utils import plot_gp
from src.utils.seeds import fix_random_seed
from src.utils.time_utils import timestamp


def experiment(
    alg: str = "gp",
    dataset_file: str = "models/2022_07_15__14_57_42/SAC_on_Qube-100-v0_100trajs.pkl.gz",
    n_trajectories: int = 100,  # 80% train, 20% test
    n_epochs: int = 1,  # almost no changes beyond 1 epoch oO
    lr: float = 1e-1,
    use_cuda: bool = True,
    # verbose: bool = False,
    plotting: bool = False,
    model_save_frequency: bool = 5,  # every x epochs
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
    x_train = df2torch(train_x_df).to(device)
    y_train = df2torch(train_y_df).to(device)
    x_test = df2torch(test_x_df).to(device)
    y_test = df2torch(test_y_df).to(device)
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

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=None).to(device)
    model = ExactGPModel(x_train, y_train, likelihood).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # margll only works for GaussianLikelihood and ExactGP
    margll_loss = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def log_metrics(epoch=-1):
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

        with open(os.path.join(results_dir, "metrics.txt"), "a") as f:
            with gpytorch.settings.fast_pred_var():
                post_pred = model(x_train)  # FIXME: leaks memory in GPU
                MAE, MSE, RMSE = compute_MAE_MSE_RMSE(post_pred.mean, y_train)
                logstring = f"Epoch {epoch} Train: MAE={MAE.item():.2f}, MSE={MSE.item():.2f}, RMSE={RMSE.item():.2f}"
                print("\r" + logstring, end=" ")
                f.write(logstring + "\t")
                post_pred = model(x_test)
                MAE, MSE, RMSE = compute_MAE_MSE_RMSE(post_pred.mean, y_test)
                logstring = f"Epoch {epoch} Test: MAE={MAE.item():.2f}, MSE={MSE.item():.2f}, RMSE={RMSE.item():.2f}"
                print("\r" + logstring, end=" ")
                f.write(logstring + "\n")

    trace = []
    for n in range(n_epochs + 1):
        # only full batches (for GP training)
        model.train()
        likelihood.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = -margll_loss(pred, y_train).sum()
        loss.backward()
        opt.step()

        # log metrics every epoch
        with torch.no_grad():
            model.eval()
            log_metrics(n)

        # hyperparams logging (lenghtscale, noise)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            trace.append(loss.detach().to("cpu").item())
            cur_ls = model.covar_module.base_kernel.lengthscale.detach().item()
            cur_noise = model.likelihood.noise.detach().item()
            # logprob fails with more than ~5 trajs (cholesky: cov not PSD)
            # cur_train_logprob = likelihood(model(x_train)).log_prob(y_train)
            # cur_test_logprob = likelihood(model(x_test)).log_prob(y_test)
            print(
                f"Iter {n}/{n_epochs}, Loss: {loss.detach().item():.3f}, "
                f"ls: {cur_ls:.3f}, noise: {cur_noise:.3f}, "
                # f"train logp: {cur_train_logprob:.2f}, "
                # f"test logp: {cur_test_logprob:.2f}, "
            )

        if n % model_save_frequency == 0:
            # Save the agent
            torch.save(model.state_dict(), os.path.join(results_dir, f"agent_{n}.pth"))

    # Save the agent after training
    torch.save(model.state_dict(), os.path.join(results_dir, f"agent_end.pth"))

    # clear gpu memory
    del pred, loss, opt, cur_ls, cur_noise
    torch.cuda.empty_cache()

    # print(f'GPU ALLOCATED: {torch.cuda.memory_allocated():,} B'.replace(',','_'))
    ####################################################################################################################
    # EVALUATION (on GPU, model prediction too slow otherwise)
    print("#### Evaluation & Plotting ####")
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        with gpytorch.settings.fast_pred_var():
            ## plot statistics over trajectories
            post_pred = likelihood(model(x_test))  # FIXME: leaks memory in GPU
            test_df["pred"] = post_pred.mean.detach().to("cpu").reshape(-1)
            # test_df['pred_var'] = post_pred.variance.detach().to('cpu').reshape(-1)
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
                f"{alg} ({len(train_traj_dfs)}/{len(test_traj_dfs)} episodes, {n_epochs} epochs)"
            )
            ax.legend()
            plt.savefig(
                os.path.join(results_dir, "mean-std_traj_plots__test_data_pred.png"),
                dpi=150,
            )
            # clear gpu memory
            del post_pred
            torch.cuda.empty_cache()

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
        if len(trace) > 1:
            fig_trace, ax_trace = plt.subplots()
            ax_trace.semilogy(trace, c="k")
            # ax_trace.plot(trace, '.', c='k') # dots for every data point
            ax_trace.set_xlabel("epochs")
            ax_trace.set_title("gp loss")
            plt.savefig(os.path.join(results_dir, "loss.png"), dpi=150)

    # # plot features on test dataset (sorted for plotting)
    # x_test_sorted, _ = x_test.sort(dim=0)
    # f = model.features(x_test_sorted).detach()
    # fig_features, ax_features = plt.subplots(figsize=(12, 9))
    # ax_features.plot(x_test_sorted, f[:, ::10], alpha=0.25)
    # ax_features.set_title("features on test dataset")

    if plotting:
        plt.show()

    print(f"Seed: {seed} - Took {time.time()-time_begin:.2f} seconds")
    print(f"Logs in {results_dir}")

    model.train()  # resets cache and frees memory


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
    print(f"GPU MEMORY LEAKS: {torch.cuda.memory_allocated():,} B".replace(",", "_"))
