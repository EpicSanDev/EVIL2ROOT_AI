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
        logger.info("DataManager initialized with symbols: %s", symbols)

    def get_initial_data(self, symbol):
        logger.info("Fetching initial data for symbol: %s", symbol)
        return yf.download(symbol, start="2020-01-01", end="2024-01-01")

    def update_data(self):
        logger.info("Updating data for all symbols.")
        for symbol in self.symbols:
            logger.info("Updating data for symbol: %s", symbol)
            new_data = yf.download(symbol, period='1d', interval='1m')  # Dernières minutes
            self.data[symbol] = self.data[symbol].append(new_data).drop_duplicates()
        logger.info("Data update completed.")

    def start_data_update(self, interval_minutes=5):
        logger.info("Starting data update every %d minutes.", interval_minutes)
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
        logger.info("RealTimeTrainer initialized.")

    def train_models(self):
        logger.info("Training models for all symbols.")
        for symbol, data in self.data_manager.data.items():
            logger.info("Training models for symbol: %s", symbol)
            self.trading_bot.price_model.train(data, symbol)
            self.trading_bot.risk_model.train(data, symbol)
            self.trading_bot.indicator_model.train(data, symbol)
            self.trading_bot.tp_sl_model.train(data, symbol)
        logger.info("Model training completed.")

    def start_training(self, interval_minutes=10):
        logger.info("Starting model training every %d minutes.", interval_minutes)
        schedule.every(interval_minutes).minutes.do(self.train_models)
        while True:
            schedule.run_pending()
            time.sleep(1)