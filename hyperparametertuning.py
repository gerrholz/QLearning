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

import optuna

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
        

learning_rate = 0.1
n_episodes = 3_000
eval_steps = 100
eval_episodes = 100
epsilon = 0.3

mean_evaluation_rewards = []
mean_episode_lengths = []

seed = 100
train_seed_offset = 0
eval_seed_offset = int(1e8)

max_steps = 1000

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def optimize_strategy(strategy_name):
    def objective(trial):
        # Define the hyperparameters to tune
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5, log=True)
        discount_factor = trial.suggest_float('discount_factor', 0.9, 0.999, log=True)
        epsilon = trial.suggest_float('epsilon', 0.01, 0.5, log=True)
        boltzmann_temperature = trial.suggest_float('temperature', 0.1, 10.0, log=True)
        ucb_c = trial.suggest_float('ucb_c', 0.1, 10.0, log=True)

        # Choose exploration strategy
        if strategy_name == 'greedy_epsilon':
            exploration_strategy = GreedyEpsilon(epsilon)
        elif strategy_name == 'boltzmann':
            exploration_strategy = Boltzmann(boltzmann_temperature)
        elif strategy_name == 'ucb':
            exploration_strategy = UCB(ucb_c)

        agent = TaxiAgent(
            learning_rate=learning_rate,
            exploration_strategy=exploration_strategy,
            discount_factor=discount_factor
        )

        seed_everything(seed)
        cumulative_rewards = []
        cumulative_reward = 0
        for episode in range(n_episodes):
            obs, info = env.reset(seed=(train_seed_offset + episode))
            done = False
            steps = 0
            episode_reward = 0

            while not done and steps <= max_steps:
                action = agent.get_action(obs, True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.update(obs, action, reward, terminated, next_obs)
                obs = next_obs
                done = terminated or truncated
                steps += 1

            
        eval_rewards = []
        for j in range(eval_episodes):
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

        mean_eval_reward = np.mean(eval_rewards)
        return mean_eval_reward

    # Create a study for the given strategy
    study_name = f'taxi_optimization_comulativereward_{strategy_name}'
    storage_name = f'sqlite:///{study_name}.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize')
    study.optimize(objective, n_trials=0) # Only load studies

    print(f'Best trial for {strategy_name}: {study.best_trial.params}')

    return study

greedy_epsilon_study = optimize_strategy('greedy_epsilon')
boltzmann_study = optimize_strategy('boltzmann')
ucb_study = optimize_strategy('ucb')

## Step 2:
## Now analyse the found parameters using the standard deviation and find the combination with the best deviation
## Only use the best sets from the first study with the mean reward 7.85

target_avg_reward = 7.85

greedy_trials = greedy_epsilon_study.trials
greedy_param_sets = [trial.params for trial in greedy_trials if trial.value == target_avg_reward]

boltzmann_trials = boltzmann_study.trials
boltzmann_param_sets = [trial.params for trial in boltzmann_trials if trial.value == target_avg_reward]

ucb_trials = ucb_study.trials
ucb_param_sets = [trial.params for trial in ucb_trials if trial.value == target_avg_reward]

def evaluate_param_set(param_set, strategy_name, n_eval_runs=100):

    
    # Extract parameters
    learning_rate = param_set['learning_rate']
    discount_factor = param_set['discount_factor']
    epsilon = param_set.get('epsilon', 0.1)  # Default value in case epsilon is not in param_set
    boltzmann_temperature = param_set.get('temperature', 1.0)  # Default value in case temperature is not in param_set
    ucb_c = param_set.get('ucb_c', 1.0)  # Default value in case ucb_c is not in param_set

    if strategy_name == 'greedy_epsilon':
        exploration_strategy = GreedyEpsilon(epsilon)
    elif strategy_name == 'boltzmann':
        exploration_strategy = Boltzmann(boltzmann_temperature)
    elif strategy_name == 'ucb':
        exploration_strategy = UCB(ucb_c)

    agent = TaxiAgent(
        learning_rate=learning_rate,
        exploration_strategy=exploration_strategy,
        discount_factor=discount_factor
    )

    seed_everything(seed)
    cumulative_rewards = []
    cumulative_reward = 0
    for episode in range(n_episodes):
        obs, info = env.reset(seed=(train_seed_offset + episode))
        done = False
        steps = 0
        episode_reward = 0

        while not done and steps <= max_steps:
            action = agent.get_action(obs, True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated
            steps += 1

            
            
    eval_rewards = []
    for j in range(eval_episodes):
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

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    return mean_reward, std_reward


results = []
for param_set in greedy_param_sets:
    mean_reward, std_reward = evaluate_param_set(param_set, 'greedy_epsilon')
    results.append((param_set, mean_reward, std_reward))


# Find the parameter set with the lowest standard deviation
best_param_set = min(results, key=lambda x: x[2])  # Sort by standard deviation

print(f'Best parameter set for Greedy Epsilon: {best_param_set[0]}')
print(f'Mean Reward: {best_param_set[1]}, Std Reward: {best_param_set[2]}')


results = []
for param_set in boltzmann_param_sets:
    mean_reward, std_reward = evaluate_param_set(param_set, 'boltzmann')
    results.append((param_set, mean_reward, std_reward))

# Find the parameter set with the lowest standard deviation
best_param_set = min(results, key=lambda x: x[2])  # Sort by standard deviation

print(f'Best parameter set for Boltzmann: {best_param_set[0]}')
print(f'Mean Reward: {best_param_set[1]}, Std Reward: {best_param_set[2]}')

results = []
for param_set in boltzmann_param_sets:
    mean_reward, std_reward = evaluate_param_set(param_set, 'ucb')
    results.append((param_set, mean_reward, std_reward))

# Find the parameter set with the lowest standard deviation
best_param_set = min(results, key=lambda x: x[2])  # Sort by standard deviation

print(f'Best parameter set for UCB: {best_param_set[0]}')
print(f'Mean Reward: {best_param_set[1]}, Std Reward: {best_param_set[2]}')


