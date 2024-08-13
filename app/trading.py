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
from app.model_trainer import ModelTrainer

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
            self.data[symbol] = pd.concat([self.data[symbol], new_data]).drop_duplicates()
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
        self.latest_signals = []  # Store the latest signals here
        logging.info("TradingBot initialized with models.")

    def get_latest_signals(self):
        """Returns the latest trading signals stored in the bot."""
        return self.latest_signals

    def train_all_models(self, data_manager):
        logging.info("Training models for all symbols.")
        for symbol, data in data_manager.data.items():
            self.price_model.train(data, symbol)
            self.risk_model.train(data, symbol)
            self.tp_sl_model.train(data, symbol)
            self.indicator_model.train(data, symbol)
        
        complete_message = "Training for all models has completed."
        self.telegram_bot.send_message(complete_message)
        logging.info(complete_message)

    def train_single_model(self, data, symbol):
        try:
            logging.info(f"Training model for {symbol}...")
            self.price_model.train(data, symbol)
            self.risk_model.train(data, symbol)
            self.tp_sl_model.train(data, symbol)
            self.indicator_model.train(data, symbol)

            single_complete_message = f"Training for {symbol} model completed."
            self.telegram_bot.send_message(single_complete_message)
            logging.info(single_complete_message)
        except Exception as e:
            error_message = f"Error during training of {symbol}: {e}"
            self.telegram_bot.send_message(error_message)
            logging.error(error_message)

    def run_reinforcement_learning(self, data_path):
        logging.info("Starting reinforcement learning training...")
        data = pd.read_csv(data_path)
        self.rl_model = train_rl_agent(data)

    def run_sentiment_analysis(self, headlines):
        return analyze_headlines(headlines)

    def run_backtest(self, data_path):
        run_backtest(data_path)

    def execute_trades(self, data_manager):
        for symbol, data in data_manager.data.items():
            predicted_price = self.price_model.predict(data, symbol)
            indicator_signal = self.indicator_model.predict(data, symbol)
            risk_decision = self.risk_model.predict(data, symbol)
            tp, sl = self.tp_sl_model.predict(data, symbol)
            rl_decision = self.rl_model.predict(data) if self.rl_model else None

            headlines = ["Example headline 1", "Example headline 2"]
            sentiment_score = np.mean(self.run_sentiment_analysis(headlines))

            decision = self.combine_signals(predicted_price, indicator_signal, risk_decision, tp, sl, rl_decision, sentiment_score)
            self.execute_trade(decision, symbol, predicted_price, tp, sl)

            self.latest_signals.append({
                'symbol': symbol,
                'decision': decision,
                'predicted_price': predicted_price,
                'tp': tp,
                'sl': sl,
                'sentiment_score': sentiment_score
            })

    def combine_signals(self, predicted_price, indicator_signal, risk_decision, tp, sl, rl_decision, sentiment_score):
        try:
            if risk_decision > 0.7:
                return "hold"

            if sentiment_score > 0.5 and rl_decision == 1 and predicted_price > indicator_signal:
                return "buy"
            elif sentiment_score < -0.5 and rl_decision == 2 and predicted_price < indicator_signal:
                return "sell"
            else:
                if tp > sl:
                    return "buy"
                elif sl > tp:
                    return "sell"
                else:
                    return "hold"
        except Exception as e:
            logging.error(f"Error in combine_signals: {e}")
            return "hold"

    def execute_trade(self, decision, symbol, predicted_price, tp, sl):
        message = f"Trading Decision for {symbol}:\n"
        message += f"Action: {decision}\n"
        message += f"Entry Price: {predicted_price}\n"
        message += f"Take Profit (TP): {tp}\n"
        message += f"Stop Loss (SL): {sl}\n"

        if decision == "buy":
            logging.info(f"Buying {symbol}")
            message += "\nAction taken: Buying"
        elif decision == "sell":
            logging.info(f"Selling {symbol}")
            message += "\nAction taken: Selling"
        else:
            logging.info(f"Holding {symbol}")
            message += "\nAction taken: Holding"
        
        try:
            self.telegram_bot.send_message(message)
        except Exception as e:
            logging.error(f"Failed to send message via Telegram: {e}")

    def start_real_time_scanning(self, data_manager, interval_seconds=60):
        """Scans the market for trading opportunities at regular intervals."""
        def scan_and_trade():
            logging.info("Scanning market for opportunities...")
            self.execute_trades(data_manager)
        
        logging.info(f"Starting real-time market scanning every {interval_seconds} seconds.")
        schedule.every(interval_seconds).seconds.do(scan_and_trade)
        
        while True:
            schedule.run_pending()
            time.sleep(1)

# Example of running backtest and trades
if __name__ == "__main__":
    data_manager = DataManager(["AAPL", "GOOGL"])  # Example symbols
    trading_bot = TradingBot()
    trading_bot.train_all_models(data_manager)
    trading_bot.run_reinforcement_learning('market_data_cleaned_auto.csv')
    
    # Running backtest
    trading_bot.run_backtest('market_data_cleaned_auto.csv')

    # Start real-time market scanning
    trading_bot.start_real_time_scanning(data_manager)