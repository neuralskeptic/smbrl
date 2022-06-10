import gym
import numpy as np
import wandb
from quanser_robots import GentlyTerminating
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from library.utils import ProgressBarManager

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 500_000,
    "env_name": "Qube-100-v0",
}
run = wandb.init(
    project="smbrl_sb3PPO_qube100",
    entity="showmezeplozz",  # needed ?
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(GentlyTerminating(env))  # record stats such as returns
    return env


env = make_env()
# env = GentlyTerminating(gym.make("Qube-100-v0"))
# env.reset()

model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")


model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()


# # Random Agent, before training
# # mean_reward_before_train = evaluate(model, num_episodes=100)
# # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# # evaluate_policy(model, env, render=True, n_eval_episodes=1)
# # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# # Train
# N_EPISODES = 10
# N_TIMESTEPS = 100_000
# N_TIMESTEPS_EFFECTIVE = (N_TIMESTEPS//model.n_steps + 1)*model.n_steps
# with ProgressBarManager(N_TIMESTEPS_EFFECTIVE) as cb:
#     model.learn(total_timesteps=N_TIMESTEPS, callback=cb)
#     # for i in range(5):
#     #     model.learn(total_timesteps=N_TIMESTEPS, callback=cb)
#     #     mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
#     #     print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}\n")

# # Evaluate the trained agent
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# evaluate_policy(model, env, render=True, n_eval_episodes=1)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}\n")


# # model.save("./models/PPO_tutorial")
# # loaded_model = PPO.load("./models/PPO_tutorial")
