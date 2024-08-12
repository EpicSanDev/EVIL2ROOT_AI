import yfinance as yf
import schedule
import time
import logging
from .models.price_prediction import PricePredictionModel
from .models.risk_management import RiskManagementModel
from .models.indicator_management import IndicatorManagementModel
from .models.tp_sl_management import TpSlManagementModel
from .telegram_bot import TelegramBot
# Configuration du logging
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
            new_data = yf.download(symbol, period='1d', interval='1m')  # Dernières minutes
            self.data[symbol] = self.data[symbol].append(new_data).drop_duplicates()
        logging.info("Data update completed.")
    
    def start_data_update(self, interval_minutes=5):
        logging.info("Starting data update every %d minutes.", interval_minutes)
        schedule.every(interval_minutes).minutes.do(self.update_data)
        while True:
            schedule.run_pending()
            time.sleep(1)

class TradingBot:
    def __init__(self):
        self.price_model = PricePredictionModel()
        self.risk_model = RiskManagementModel()
        self.indicator_model = IndicatorManagementModel()
        self.tp_sl_model = TpSlManagementModel()
        self.telegram_bot = TelegramBot()  # Initialiser le bot Telegram
        logging.info("TradingBot initialized with models.")

    def get_trading_decisions(self, data_manager):
        logging.info("Generating trading decisions.")
        decisions = {}
        for symbol, data in data_manager.data.items():
            logging.debug("Processing symbol: %s", symbol)
            predicted_price = self.price_model.predict(data, symbol)
            predicted_risk = self.risk_model.predict(data, symbol)
            adjusted_indicator = self.indicator_model.predict(data, symbol)
            predicted_tp, predicted_sl = self.tp_sl_model.predict(data, symbol)

            # Logique d'achat
            if (predicted_price > data['Close'][-1] and
                    predicted_risk < 0.02 and
                    adjusted_indicator > data['Close'][-1]):
                decision = "Acheter"
                tp, sl = predicted_tp, predicted_sl
                message = f"Buy decision for {symbol}: TP={tp}, SL={sl}"
                logging.info(message)
                self.telegram_bot.send_message(message)

            # Logique de vente
            elif (predicted_price < data['Close'][-1] and
                  predicted_risk < 0.02 and
                  adjusted_indicator < data['Close'][-1]):
                decision = "Vendre"
                tp, sl = predicted_tp, predicted_sl
                message = f"Sell decision for {symbol}: TP={tp}, SL={sl}"
                logging.info(message)
                self.telegram_bot.send_message(message)

            # Ne rien faire
            else:
                decision = "Ne rien faire"
                tp, sl = None, None
                logging.info(f"No action for {symbol}")

            decisions[symbol] = {
                "décision": decision,
                "indicateur ajusté": adjusted_indicator,
                "Take Profit": tp,
                "Stop Loss": sl
            }
        return decisions

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

from concurrent.futures import ThreadPoolExecutor
import logging

class ModelTrainer:
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot

    def train_all_models(self, data_manager):
        with ThreadPoolExecutor() as executor:
            futures = []
            for symbol, data in data_manager.data.items():
                futures.append(executor.submit(self.trading_bot.price_model.train, data, symbol))
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error training model: {e}")

import gym
from gym import spaces
import numpy as np

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
        if self.current_step >= len(self.data):
            done = True
        else:
            done = False

        # Calculer la récompense en fonction de l'action prise
        reward = 0
        if action == 1:  # Buy
            reward = self.data.iloc[self.current_step]['Close'] - self.data.iloc[self.current_step - 1]['Close']
        elif action == 2:  # Sell
            reward = self.data.iloc[self.current_step - 1]['Close'] - self.data.iloc[self.current_step]['Close']

        obs = self.data.iloc[self.current_step].values
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass
# app/trading.py
from app.models.rl_trading import train_rl_agent
from app.models.sentiment_analysis import analyze_headlines
from app.models.backtesting import run_backtest
import pandas as pd

class TradingBot:
    def __init__(self):
        # Initialisation des modèles et des composants ici
        self.price_model = PricePredictionModel()
        self.risk_model = RiskManagementModel()
        self.tp_sl_model = TpSlManagementModel()

    def run_reinforcement_learning(self, data_path):
        data = pd.read_csv(data_path)
        model = train_rl_agent(data)
        return model

    def run_sentiment_analysis(self, headlines):
        sentiments = analyze_headlines(headlines)
        return sentiments

    def run_backtest(self, data_path):
        run_backtest(data_path)