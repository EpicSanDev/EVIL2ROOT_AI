import yfinance as yf
import schedule
import time
from .models.price_prediction import PricePredictionModel
from .models.risk_management import RiskManagementModel
from .models.indicator_management import IndicatorManagementModel
from .models.tp_sl_management import TpSlManagementModel

class DataManager:
    def __init__(self, symbols):
        self.symbols = symbols
        self.data = {symbol: self.get_initial_data(symbol) for symbol in symbols}

    def get_initial_data(self, symbol):
        return yf.download(symbol, start="2020-01-01", end="2024-01-01")

    def update_data(self):
        for symbol in self.symbols:
            new_data = yf.download(symbol, period='1d', interval='1m')  # Dernières minutes
            self.data[symbol] = self.data[symbol].append(new_data).drop_duplicates()

    def start_data_update(self, interval_minutes=5):
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

    def get_trading_decisions(self, data_manager):
        decisions = {}
        for symbol, data in data_manager.data.items():
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

            # Logique de vente
            elif (predicted_price < data['Close'][-1] and
                  predicted_risk < 0.02 and
                  adjusted_indicator < data['Close'][-1]):
                decision = "Vendre"
                tp, sl = predicted_tp, predicted_sl

            # Ne rien faire
            else:
                decision = "Ne rien faire"
                tp, sl = None, None

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

    def train_models(self):
        for symbol, data in self.data_manager.data.items():
            self.trading_bot.price_model.train(data, symbol)
            self.trading_bot.risk_model.train(data, symbol)
            self.trading_bot.indicator_model.train(data, symbol)
            self.trading_bot.tp_sl_model.train(data, symbol)

    def start_training(self, interval_minutes=10):
        schedule.every(interval_minutes).minutes.do(self.train_models)
        while True:
            schedule.run_pending()
            time.sleep(1)

    def get_server_stats(self):
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent
        }