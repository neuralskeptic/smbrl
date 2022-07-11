import gym
import numpy as np
from quanser_robots import GentlyTerminating
from stable_baselines3 import PPO

env = GentlyTerminating(gym.make("CartpoleStabLong-v0"))

model = PPO("MlpPolicy", env, verbose=0)


from stable_baselines3.common.evaluation import evaluate_policy


# same as
def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            env.render()
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


# Random Agent, before training
# mean_reward_before_train = evaluate(model, num_episodes=100)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# evaluate_policy(model, env, render=True, n_eval_episodes=1)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent for 10000 steps
model.learn(total_timesteps=10000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
evaluate_policy(model, env, render=True, n_eval_episodes=1)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")