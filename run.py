import asyncio
from app.trading import TradingBot, DataManager, RealTimeTrainer
from app.telegram_bot import TelegramBot
import schedule
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Exemple de symboles combinés pour les actions et le Forex
forex_symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
symbols = stock_symbols + forex_symbols

# Initialisation des composants
data_manager = DataManager(symbols)
trading_bot = TradingBot()
telegram_bot = TelegramBot()
trainer = RealTimeTrainer(data_manager, trading_bot)

def execute_trades(trading_bot, data_manager):
    logger.info("Executing trades...")
    decisions = trading_bot.get_trading_decisions(data_manager)
    for symbol, decision_data in decisions.items():
        logger.info(f"Symbol: {symbol}, Decision: {decision_data['décision']}, "
                    f"TP: {decision_data['Take Profit']}, SL: {decision_data['Stop Loss']}")

schedule.every(1).minutes.do(execute_trades, trading_bot, data_manager)

async def main():
    start_message = "Trading bot has started successfully."
    logger.info(start_message)

    try:
        await telegram_bot.send_message(start_message)
        logger.info("Start message sent successfully via Telegram.")
    except Exception as e:
        logger.error(f"Failed to send start message via Telegram: {e}")
    
    logger.info("Starting model training...")
    trainer.train_models()

    logger.info("Starting data update and trading bot...")
    data_manager.start_data_update(interval_minutes=5)
    
    while True:
        schedule.run_pending()
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())