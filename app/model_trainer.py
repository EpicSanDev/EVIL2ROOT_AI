import logging
from concurrent.futures import ThreadPoolExecutor

class ModelTrainer:
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot

    def train_all_models(self, data_manager):
        try:
            # Notifier que l'entraînement commence
            start_message = "Training for all models is starting..."
            self.trading_bot.telegram_bot.send_message(start_message)
            logging.info(start_message)

            with ThreadPoolExecutor() as executor:
                futures = []
                for symbol, data in data_manager.data.items():
                    futures.append(executor.submit(self.train_single_model, data, symbol))

                # Attendre que tous les modèles soient entraînés
                for future in futures:
                    future.result()

            # Notifier que l'entraînement est terminé
            complete_message = "Training for all models has completed."
            self.trading_bot.telegram_bot.send_message(complete_message)
            logging.info(complete_message)

        except Exception as e:
            error_message = f"Error during model training: {e}"
            self.trading_bot.telegram_bot.send_message(error_message)
            logging.error(error_message)

    def train_single_model(self, data, symbol):
        try:
            logging.info(f"Training model for {symbol}...")
            self.trading_bot.price_model.train(data, symbol)
            self.trading_bot.risk_model.train(data, symbol)
            self.trading_bot.tp_sl_model.train(data, symbol)
            self.trading_bot.indicator_model.train(data, symbol)

            # Notifier que l'entraînement pour un symbole est terminé
            single_complete_message = f"Training for {symbol} model completed."
            self.trading_bot.telegram_bot.send_message(single_complete_message)
            logging.info(single_complete_message)

        except Exception as e:
            error_message = f"Error during training of {symbol}: {e}"
            self.trading_bot.telegram_bot.send_message(error_message)
            logging.error(error_message)