## This script trains 50 agents with the optimal results from the hyperparametertuning file

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

import mlflow

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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_and_evaluate(strategy_name, strategy_params, n_agents=50, n_episodes=3000, eval_episodes=1000, max_steps=1000, seed_base=1000):
    results = {
        "average_rewards": [],
        "cumulative_rewards": [],
        "std_rewards": [],
        "convergence_rates": [],
        "episodic_rewards": [],
        "success_rates": [],
        "q_tables": [],
        "seeds": []
    }

    for i in tqdm(range(n_agents), desc=f"Training agents for {strategy_name}"):
        seed = seed_base + i
        seed_everything(seed)

        if strategy_name == 'greedy_epsilon':
            exploration_strategy = GreedyEpsilon(strategy_params['epsilon'])
        elif strategy_name == 'boltzmann':
            exploration_strategy = Boltzmann(strategy_params['temperature'])
        elif strategy_name == 'ucb':
            exploration_strategy = UCB(strategy_params['ucb_c'])

        agent = TaxiAgent(
            learning_rate=strategy_params['learning_rate'],
            exploration_strategy=exploration_strategy,
            discount_factor=strategy_params['discount_factor']
        )

        cumulative_rewards = []
        episodic_rewards = []
        for episode in range(n_episodes):
            obs, info = env.reset(seed=seed + episode)
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
                episode_reward += reward
            cumulative_rewards.append(episode_reward)
            episodic_rewards.append(episode_reward)

        # Calculate convergence rate
        if len(cumulative_rewards) > 1:
            convergence_rate = np.mean(np.diff(cumulative_rewards[-100:]))
        else:
            convergence_rate = 0

        # Evaluation
        eval_rewards = []
        success_count = 0
        for j in range(eval_episodes):
            obs, info = env.reset(seed=seed + n_episodes + j)
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
            if done and sum_reward > 0:
                success_count += 1

        mean_eval_reward = np.mean(eval_rewards)
        std_eval_reward = np.std(eval_rewards)
        success_rate = success_count / eval_episodes

        results["average_rewards"].append(mean_eval_reward)
        results["cumulative_rewards"].append(sum(cumulative_rewards))
        results["std_rewards"].append(std_eval_reward)
        results["convergence_rates"].append(convergence_rate)
        results["episodic_rewards"].append(episodic_rewards)
        results["success_rates"].append(success_rate)
        results["q_tables"].append(dict(agent.q_values))
        results["seeds"].append(seed)

    return results

def log_results(strategy_name, strategy_params, results):
    with mlflow.start_run(run_name=strategy_name):
        mlflow.log_params(strategy_params)

        for metric, values in results.items():
            if metric == "episodic_rewards":
                for idx, episodic_reward in enumerate(values):
                    mlflow.log_metric(f"{metric}_agent_{idx}", sum(episodic_reward))
            elif metric == "q_tables":
                for idx, q_table in enumerate(values):
                    mlflow.log_dict(q_table, f"{strategy_name}_agent_{idx}_q_table.json")
            elif metric == "seeds":
                for idx, seed in enumerate(values):
                    mlflow.log_param(f"seed_agent_{idx}", seed)
            else:
                mlflow.log_metric(f"{metric}_mean", np.mean(values))
                mlflow.log_metric(f"{metric}_std", np.std(values))
                mlflow.log_metric(f"{metric}_median", np.median(values))
                mlflow.log_metric(f"{metric}_max", np.max(values))
                mlflow.log_metric(f"{metric}_min", np.min(values))

        for idx, avg_reward in enumerate(results["average_rewards"]):
            mlflow.log_metric(f"average_reward_agent_{idx}", avg_reward)

        for idx, avg_reward in enumerate(results["success_rates"]):
            mlflow.log_metric(f"success_rates_agent_{idx}", avg_reward)


if __name__ == "__main__":
    # Define the best parameters for each strategy (example values, replace with actual best parameters)
    best_params_greedy_epsilon = {
        "learning_rate": 0.2344206791710185,
        "discount_factor": 0.9176983760342422,
        "epsilon": 0.03820245426454189
    }

    best_params_boltzmann = {
        "learning_rate": 0.19715043101673577,
        "discount_factor": 0.9027741457357016,
        "temperature": 0.11816915703915966
    }

    best_params_ucb = {
        "learning_rate": 0.3295884549251193,
        "discount_factor": 0.9615733052900906,
        "ucb_c": 0.8476924674234931
    }

    # Train and evaluate agents for each strategy
    results_greedy_epsilon = train_and_evaluate("greedy_epsilon", best_params_greedy_epsilon)
    results_boltzmann = train_and_evaluate("boltzmann", best_params_boltzmann)
    results_ucb = train_and_evaluate("ucb", best_params_ucb)

    # Log results using MLflow
    log_results("greedy_epsilon", best_params_greedy_epsilon, results_greedy_epsilon)
    log_results("boltzmann", best_params_boltzmann, results_boltzmann)
    log_results("ucb", best_params_ucb, results_ucb)
