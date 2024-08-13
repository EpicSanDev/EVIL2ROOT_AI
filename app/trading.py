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
        return yf.download(symbol, start="2001-01-01", end="2024-01-01")

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

class TradingBot:
    def __init__(self):
        self.price_model = PricePredictionModel()
        self.risk_model = RiskManagementModel()
        self.tp_sl_model = TpSlManagementModel()
        self.indicator_model = IndicatorManagementModel()
        self.rl_model = None  # For RL agent, to be trained later
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
        try:
            # Logic to combine model predictions into a final trading decision
            if risk_decision > 0.7:  # High risk
                logging.info(f"High risk detected for symbol, holding position.")
                return "hold"

            if rl_decision is not None:  # Reinforcement Learning decision takes priority
                if rl_decision == 1:
                    return "buy"
                elif rl_decision == 2:
                    return "sell"

            if sentiment_score > 0.5 and predicted_price > indicator_signal:
                return "buy"
            elif sentiment_score < -0.5 and predicted_price < indicator_signal:
                return "sell"

            # If none of the above are decisive, fall back to TP/SL management
            if tp > sl:
                return "buy"
            elif sl > tp:
                return "sell"

            return "hold"  # Default to holding if signals are conflicting or neutral

        except Exception as e:
            logging.error(f"Error in combine_signals: {e}")
            return "hold"

    def execute_trade(self, decision, symbol):
        if decision == "buy":
            logging.info(f"Buying {symbol}")
        elif decision == "sell":
            logging.info(f"Selling {symbol}")
        else:
            logging.info(f"Holding {symbol}")

# Example of running backtest and trades
if __name__ == "__main__":
    data_manager = DataManager(["AAPL", "GOOGL"])  # Example symbols
    trading_bot = TradingBot()
    trading_bot.train_all_models(data_manager)
    trading_bot.run_reinforcement_learning('market_data_cleaned_auto.csv')
    
    # Running backtest
    trading_bot.run_backtest('market_data_cleaned_auto.csv')

    # Execute trades
    trading_bot.execute_trades(data_manager)