from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import random

import gymnasium as gym

env = gym.make("Taxi-v3")

class TaxiAgent:

    def __init__(self, learning_rate: float, exploration_strategy, discount_factor: float) -> None:
        self.env = env
        # Empty dictionary of state-action values
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.learning_rate = learning_rate
        self.exploration_strategy = exploration_strategy
        self.discount_factor = discount_factor

        self.training_error = []

    def get_action(self, obs):
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
        

learning_rate = 0.001
n_episodes = 100_000
epsilon = 0.1

greedy_epsilon = GreedyEpsilon(epsilon)
boltzmann = Boltzmann(200)

agent = TaxiAgent(
    learning_rate=learning_rate,
    exploration_strategy=boltzmann,
    discount_factor=0.7
)

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()