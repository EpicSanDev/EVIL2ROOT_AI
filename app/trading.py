from gym import Env
from gym.spaces import Discrete, Box
import yfinance as yf
import schedule
import time
import logging
import pandas as pd
import numpy as np
from app.models.price_prediction import PricePredictionModel
from app.models.risk_management import RiskManagementModel
from app.models.tp_sl_management import TpSlManagementModel
from app.models.indicator_management import IndicatorManagementModel
from app.models.rl_trading import train_rl_agent
from app.models.sentiment_analysis import analyze_headlines
from app.models.backtesting import run_backtest
from app.telegram_bot import TelegramBot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, symbols):
        self.symbols = symbols
        self.data = {symbol: self.get_initial_data(symbol) for symbol in symbols}
        logging.info("DataManager initialized with symbols: %s", symbols)

    def get_initial_data(self, symbol):
        logging.info("Fetching initial data for symbol: %s", symbol)
        return yf.download(symbol, start="2020-01-01", end="2024-01-01")

    def update_data(self):
        logging.info("Updating data for all symbols.")
        for symbol in self.symbols:
            logging.info("Updating data for symbol: %s", symbol)
            new_data = yf.download(symbol, period='1d', interval='1m')
            self.data[symbol] = self.data[symbol].append(new_data).drop_duplicates()
        logging.info("Data update completed.")
    
    def start_data_update(self, interval_minutes=5):
        logging.info("Starting data update every %d minutes.", interval_minutes)
        schedule.every(interval_minutes).minutes.do(self.update_data)
        while True:
            schedule.run_pending()
            time.sleep(1)

class RealTimeTrainer:
    def __init__(self, data_manager, trading_bot):
        self.data_manager = data_manager
        self.trading_bot = trading_bot
        logging.info("RealTimeTrainer initialized.")

    def train_models(self):
        logging.info("Training models for all symbols.")
        for symbol, data in self.data_manager.data.items():
            if len(data) < 60:
                logging.warning(f"Not enough data to train models for symbol: {symbol}")
                continue
            self.trading_bot.price_model.train(data, symbol)
            self.trading_bot.risk_model.train(data, symbol)
            self.trading_bot.indicator_model.train(data, symbol)
            self.trading_bot.tp_sl_model.train(data, symbol)
        logging.info("Model training completed.")

    def start_training(self, interval_minutes=10):
        logger.info("Starting model training every %d minutes.", interval_minutes)
        schedule.every(interval_minutes).minutes.do(self.train_models)
        while True:
            schedule.run_pending()
            time.sleep(1)

class TradingBot:
    def __init__(self):
        self.price_model = PricePredictionModel()
        self.risk_model = RiskManagementModel()
        self.tp_sl_model = TpSlManagementModel()
        self.indicator_model = IndicatorManagementModel()
        self.rl_model = None  # Pour l'agent RL, que nous allons former plus tard
        self.telegram_bot = TelegramBot()
        logging.info("TradingBot initialized with models.")

    def train_all_models(self, data_manager):
        logging.info("Training models for all symbols.")
        for symbol, data in data_manager.data.items():
            self.price_model.train(data, symbol)
            self.risk_model.train(data, symbol)
            self.tp_sl_model.train(data, symbol)
            self.indicator_model.train(data, symbol)
        logging.info("Model training completed.")

    def run_reinforcement_learning(self, data_path):
        logging.info("Starting reinforcement learning training...")
        data = pd.read_csv(data_path)
        self.rl_model = train_rl_agent(data)

    def run_sentiment_analysis(self, headlines):
        sentiments = analyze_headlines(headlines)
        return sentiments

    def run_backtest(self, data_path):
        run_backtest(data_path)

    def execute_trades(self, data_manager):
        for symbol, data in data_manager.data.items():
            predicted_price = self.price_model.predict(data, symbol)
            indicator_signal = self.indicator_model.predict(data, symbol)
            risk_decision = self.risk_model.predict(data, symbol)
            tp, sl = self.tp_sl_model.predict(data, symbol)
            rl_decision = None
            if self.rl_model:
                rl_decision = self.rl_model.predict(data)  # Adjust based on how RL is used

            headlines = ["Example headline 1", "Example headline 2"]
            sentiments = self.run_sentiment_analysis(headlines)
            sentiment_score = sum(sentiments) / len(sentiments)

            decision = self.combine_signals(predicted_price, indicator_signal, risk_decision, tp, sl, rl_decision, sentiment_score)
            self.execute_trade(decision, symbol)

    def combine_signals(self, predicted_price, indicator_signal, risk_decision, tp, sl, rl_decision, sentiment_score):
        if sentiment_score > 0.5 and rl_decision == 1:
            return "buy"
        elif sentiment_score < -0.5 and rl_decision == 2:
            return "sell"
        else:
            return "hold"

    def execute_trade(self, decision, symbol):
        if decision == "buy":
            logger.info(f"Buying {symbol}")
        elif decision == "sell":
            logger.info(f"Selling {symbol}")
        else:
            logger.info(f"Holding {symbol}")



class TradingEnv(Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0

        # Vérifiez que les données ont bien des colonnes numériques
        print("Initial data shape:", self.data.shape)
        assert self.data.shape[1] > 0, "Data must have more than 0 columns"

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