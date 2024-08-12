# app/models/rl_trading.py
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN

class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0

        # Définir les espaces d'actions et d'observations
        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns),), dtype=np.float32)

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
    env = TradingEnv(data)
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("dqn_trading_model")
    return model