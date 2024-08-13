import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from gym import Env
from gym.spaces import Discrete, Box
import logging

# Votre environnement de trading personnalisé
class TradingEnv(Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0

        # Vérifiez que les données ont bien des colonnes numériques
        print("Initial data columns:", self.data.columns)
        print("Initial data types:", self.data.dtypes)
        assert self.data.shape[1] > 0, "Data must have more than 0 columns"
        assert all(self.data.dtypes.apply(lambda x: np.issubdtype(x, np.number))), "All columns must be numeric"

        # Assurez-vous que l'espace d'observation a les bonnes dimensions
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32)
        self.action_space = Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell

    def reset(self):
        self.current_step = 0
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        print(f"Reset Observation shape: {obs.shape}")
        assert obs.shape == self.observation_space.shape, f"Expected shape {self.observation_space.shape}, but got {obs.shape}"
        return obs

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = 0
        if action == 1:  # Buy
            reward = self.data.iloc[self.current_step]['Close'] - self.data.iloc[self.current_step - 1]['Close']
        elif action == 2:  # Sell
            reward = self.data.iloc[self.current_step - 1]['Close'] - self.data.iloc[self.current_step]['Close']
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        
        # Vérifiez que les dimensions de obs sont correctes
        print(f"Step Observation shape: {obs.shape}")
        assert obs.shape == self.observation_space.shape, f"Expected shape {self.observation_space.shape}, but got {obs.shape}"

        return obs, reward, done, {}

    def render(self, mode='human'):
        pass
    

def train_rl_agent(data):
    # Préparer les données : retirer les colonnes non numériques
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Ajouter des instructions de journalisation pour vérifier les colonnes des données
    logging.info("Colonnes des données chargées : %s", data.columns)
    logging.info("Colonnes numériques des données : %s", numeric_data.columns)
    
    # Vérifier que les données contiennent des colonnes numériques
    if numeric_data.shape[1] == 0:
        raise ValueError("Les données doivent contenir au moins une colonne numérique.")
    
    # Créer l'environnement avec les données numériques
    env = TradingEnv(numeric_data)

    # Entraîner le modèle
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    return model


def run_reinforcement_learning(self, data_path):
    logging.info("Starting reinforcement learning training...")
    data = pd.read_csv(data_path)
    
    # Ajouter des instructions de journalisation pour vérifier les données chargées
    logging.info("Données chargées depuis %s : %s", data_path, data.head())
    
    self.rl_model = train_rl_agent(data)
