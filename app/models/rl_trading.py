# app/models/rl_trading.py

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from gym import Env
from gym.spaces import Discrete, Box

# Votre environnement de trading personnalisé
class TradingEnv(Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0

        # Définir les espaces d'actions et d'observations
        self.action_space = Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data)
        reward = 0
        if action == 1:  # Buy
            reward = self.data.iloc[self.current_step]['Close'] - self.data.iloc[self.current_step - 1]['Close']
        elif action == 2:  # Sell
            reward = self.data.iloc[self.current_step - 1]['Close'] - self.data.iloc[self.current_step]['Close']
        obs = self.data.iloc[self.current_step].values
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

def train_rl_agent(data):
    # Préparer les données : retirer les colonnes non numériques
    numeric_data = data.select_dtypes(include=[np.number])

    # Créer l'environnement avec les données numériques
    env = TradingEnv(numeric_data)

    # Entraîner le modèle
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    return model