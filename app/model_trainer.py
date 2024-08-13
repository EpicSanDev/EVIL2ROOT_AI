import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

class ModelTrainer:
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot

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
            self.trading_bot.price_model.train(data, symbol)
            self.trading_bot.risk_model.train(data, symbol)
            self.trading_bot.tp_sl_model.train(data, symbol)
            self.trading_bot.indicator_model.train(data, symbol)

            # Notify that training for a symbol is complete
            single_complete_message = f"Training for {symbol} model completed."
            asyncio.run(self.trading_bot.telegram_bot.send_message(single_complete_message))
            logging.info(single_complete_message)

        except Exception as e:
            error_message = f"Error during training of {symbol}: {e}"
            asyncio.run(self.trading_bot.telegram_bot.send_message(error_message))
            logging.error(error_message)
