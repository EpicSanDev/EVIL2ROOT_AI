import logging
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio
import joblib  # For saving and loading models
from keras.models import load_model

class ModelTrainer:
    def __init__(self, trading_bot, model_dir="saved_models"):
        self.trading_bot = trading_bot
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)  # Create directory if it doesn't exist

    def train_or_load_model(self, data, symbol):
        model_path = f'models/{symbol}_model.h5'
        if os.path.exists(model_path):
            logging.info(f"Loading existing model for {symbol}.")
            model = load_model(model_path)
        else:
            logging.info(f"Training new model for {symbol}.")
            model = self.train_model(data, symbol)
            model.save(model_path)
        return model

    async def train_all_models(self, data_manager):
        try:
            # Notify that training is starting
            start_message = "Training for all models is starting..."
            await self.trading_bot.telegram_bot.send_message(start_message)
            logging.info(start_message)

            with ThreadPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(executor, self.train_single_model, data, symbol)
                    for symbol, data in data_manager.data.items()
                ]

                # Wait for all models to be trained
                await asyncio.gather(*futures)

            # Notify that training is complete
            complete_message = "Training for all models has completed."
            await self.trading_bot.telegram_bot.send_message(complete_message)
            logging.info(complete_message)

        except Exception as e:
            error_message = f"Error during model training: {e}"
            await self.trading_bot.telegram_bot.send_message(error_message)
            logging.error(error_message)

    def train_single_model(self, data, symbol):
        try:
            logging.info(f"Training model for {symbol}...")

            # Define file paths for each model
            price_model_path = os.path.join(self.model_dir, f"{symbol}_price_model.pkl")
            risk_model_path = os.path.join(self.model_dir, f"{symbol}_risk_model.pkl")
            tp_sl_model_path = os.path.join(self.model_dir, f"{symbol}_tp_sl_model.pkl")
            indicator_model_path = os.path.join(self.model_dir, f"{symbol}_indicator_model.pkl")

            # Check if models already exist
            if os.path.exists(price_model_path) and os.path.exists(risk_model_path) \
                    and os.path.exists(tp_sl_model_path) and os.path.exists(indicator_model_path):
                logging.info(f"Models for {symbol} already exist. Skipping training.")
            else:
                # Train and save the models
                self.trading_bot.price_model.train(data, symbol)
                joblib.dump(self.trading_bot.price_model, price_model_path)
                
                self.trading_bot.risk_model.train(data, symbol)
                joblib.dump(self.trading_bot.risk_model, risk_model_path)
                
                self.trading_bot.tp_sl_model.train(data, symbol)
                joblib.dump(self.trading_bot.tp_sl_model, tp_sl_model_path)
                
                self.trading_bot.indicator_model.train(data, symbol)
                joblib.dump(self.trading_bot.indicator_model, indicator_model_path)

            # Notify that training for a symbol is complete
            single_complete_message = f"Training for {symbol} model completed."
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(self.trading_bot.telegram_bot.send_message(single_complete_message), loop)
            logging.info(single_complete_message)

        except Exception as e:
            error_message = f"Error during training of {symbol}: {e}"
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(self.trading_bot.telegram_bot.send_message(error_message), loop)
            logging.error(error_message)