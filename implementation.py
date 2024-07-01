from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import random

import gymnasium as gym

import os

env = gym.make("Taxi-v3").env

class TaxiAgent:

    def __init__(self, learning_rate: float, exploration_strategy, discount_factor: float) -> None:
        self.env = env
        # Empty dictionary of state-action values
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.learning_rate = learning_rate
        self.exploration_strategy = exploration_strategy
        self.discount_factor = discount_factor

        self.training_error = []

    def get_action(self, obs, train):
        if not train:
            return int(np.argmax(self.q_values[obs]))
        return self.exploration_strategy.get_action(obs, self.q_values)

    def update(self, obs, action, reward, terminated, next_obs):

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[obs][action])

        self.q_values[obs][action] += self.learning_rate * temporal_difference

        self.training_error.append(temporal_difference)


class GreedyEpsilon:

    def __init__(self, epsilon) -> None:
        self.epsilon = epsilon

    def get_action(self, obs, q_values):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        
        else:
            return int(np.argmax(q_values[obs]))
        
class Boltzmann:

    def __init__(self, T) -> None:
        self.T = T

    def get_action(self, obs, q_values):

        actions = [i for i in range(env.action_space.n)]

        probs = []

        exp_values = []
        for act in actions:
            exp_values.append(np.exp(q_values[obs][act]/self.T))
        
        dem = sum(exp_values)


        for act in actions:
            num = np.exp(q_values[obs][act]/self.T)

            probs.append(round(num/dem, 10))

        return random.choices(actions, weights=probs, k=1)[0]
    
class UCB:

    def __init__(self, c) -> None:
        self.c = c
        self.action_counts = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs, q_values):
        total_counts = sum(self.action_counts[obs]) + 1 #Avoid log with zero
        ucb_values = q_values[obs] + self.c * np.sqrt(np.log(total_counts) / (self.action_counts[obs] + 1e-5))

        action = int(np.argmax(ucb_values))
        self.action_counts[obs][action] += 1
        return action
        

learning_rate = 0.010140147623771485
n_episodes = 10_000
eval_steps = 100
eval_episodes = 100
epsilon = 0.010140147623771485

mean_evaluation_rewards = []
mean_episode_lengths = []

seed = 100
train_seed_offset = 0
eval_seed_offset = int(1e8)

greedy_epsilon = GreedyEpsilon(epsilon)
boltzmann = Boltzmann(1)
ucb = UCB(1000.0)
agent = TaxiAgent(
    learning_rate=0.49067081915529676,
    exploration_strategy=greedy_epsilon,
    discount_factor=0.9691316603842797,
)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed = 42
train_seed_offset = 0
eval_seed_offset = int(1e8)
max_steps = 1000

seed_everything(seed)
for episode in tqdm(range(n_episodes)):
    #env.seed(seed)
    obs, info = env.reset(seed=(train_seed_offset + episode))
    done = False
    sum_reward = 0
    steps = 0

    while not done and steps <= max_steps:
        action = agent.get_action(obs, True)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        sum_reward += reward
        done = terminated or truncated
        obs = next_obs
        steps += 1

    if (episode + 1) % eval_steps == 0:  # Evaluate every eval_steps episodes
        print(f'Train Episode: {episode + 1} Reward: {sum_reward} Steps: {steps}')

        eval_rewards = []
        eval_lengths = []
        for j in range(eval_episodes):
            #env.seed(eval_seed_offset + j)
            obs, info = env.reset(seed=(eval_seed_offset + j))
            sum_reward = 0
            done = False
            steps = 0

            while not done and steps <= max_steps:
                action = agent.get_action(obs, False)
                next_obs, reward, terminated, truncated, info = env.step(action)
                sum_reward += reward
                obs = next_obs
                done = terminated or truncated
                steps += 1

            eval_rewards.append(sum_reward)
            eval_lengths.append(steps)

        mean_evaluation_rewards.append(np.mean(eval_rewards))
        mean_episode_lengths.append(np.mean(eval_lengths))
        print(f'Mean Evaluation Reward after {episode + 1} episodes: {mean_evaluation_rewards[-1]}')
        print(f'Mean Evaluation Length after {episode + 1} episodes: {mean_episode_lengths[-1]}')


#rolling_length = 10
#fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
#axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
#reward_moving_average = (
#    np.convolve(
#        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
#    )
#    / rolling_length
#)
#axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
#axs[1].set_title("Episode lengths")
#length_moving_average = (
#    np.convolve(
#        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
#    )
#    / rolling_length
#)
#axs[1].plot(range(len(length_moving_average)), length_moving_average)
#axs[2].set_title("Training Error")
#training_error_moving_average = (
#    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
#    / rolling_length
#)
#axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
#plt.tight_layout()
#plt.show()

# Plot the mean evaluation rewards
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(range(eval_steps, n_episodes + 1, eval_steps), mean_evaluation_rewards)
plt.xlabel('Episodes')
plt.ylabel('Mean Evaluation Reward')
plt.title('Mean Evaluation Reward vs. Episodes')
plt.grid(True)

# Plot the mean episode lengths
plt.subplot(2, 1, 2)
plt.plot(range(eval_steps, n_episodes + 1, eval_steps), mean_episode_lengths)
plt.xlabel('Episodes')
plt.ylabel('Mean Episode Length')
plt.title('Mean Episode Length vs. Episodes')
plt.grid(True)

plt.tight_layout()
plt.show()