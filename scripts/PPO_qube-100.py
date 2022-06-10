import time

import gym
import numpy as np
import torch
from quanser_robots import GentlyTerminating
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb
from library.utils import structdict  # use dicts like classes (structs)

if __name__ == "__main__":
    ## problem, algo and hyperparams
    config = structdict(
        {
            "env_name": "Qube-100-v0",
            "policy_type": "MlpPolicy",
            "policy_kwargs": {
                "activation_fn": torch.nn.ReLU,
                "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
            },
            "seed": 1234,
            "total_timesteps": 500_000,
            "lr": 1e-4,
            "minibatch_size": 128,
            "n_epochs": 4,  # TODO correct?
            "rollouts_per_update": 10,  # TODO where to put this?
            # "env_steps_per_rollout": 2048, # defined in gym env
            "ppo_clip_range": 0.2,  # TODO correct?
        }
    )
    # set_random_seed(config["seed"])

    ## wand config & callback
    run_name = f"{config.env_name}__sb3PPO__{config.seed}__{int(time.time())}"
    run = wandb.init(
        name=run_name,
        entity="showmezeplozz",  # needed ?
        project="smbrl_sb3PPO_qube100",
        group="experiment_1",
        job_type="train",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    wandcallback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    ## create envs
    env = Monitor(GentlyTerminating(gym.make(config.env_name)))
    set_random_seed(config.seed)
    # env = make_vec_env(
    #     config.env_name,
    #     n_envs=2,
    #     seed=config.seed,
    #     vec_env_cls=DummyVecEnv,
    #     # vec_env_cls=SubprocVecEnv,
    #     wrapper_class=GentlyTerminating,
    # )

    ## create model
    model = PPO(
        policy=config.policy_type,
        env=env,
        learning_rate=config.lr,
        # n_steps=???,
        batch_size=config.minibatch_size,
        n_epochs=config.n_epochs,
        tensorboard_log=f"runs/{run.id}",
        policy_kwargs=config.policy_kwargs,
        verbose=1,
        seed=config.seed,
    )

    ## train model
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=wandcallback,
    )

    ## tell wandb we're done
    run.finish()
