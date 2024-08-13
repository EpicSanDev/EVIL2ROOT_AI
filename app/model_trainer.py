# app/model_trainer.py
import logging
from concurrent.futures import ThreadPoolExecutor
class ModelTrainer:
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot

    def train_all_models(self, data_manager):
        with ThreadPoolExecutor() as executor:
            futures = []
            for symbol, data in data_manager.data.items():
                # Assurez-vous que price_model est correctement utilisé ici
                futures.append(executor.submit(self.trading_bot.price_model.train, data, symbol))
                futures.append(executor.submit(self.trading_bot.risk_model.train, data, symbol))
                futures.append(executor.submit(self.trading_bot.tp_sl_model.train, data, symbol))
                futures.append(executor.submit(self.trading_bot.indicator_model.train, data, symbol))
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error training model: {e}")