import os
import asyncio
import yfinance as yf
import schedule
import logging
import pandas as pd
import numpy as np
from app.trading import TradingBot, DataManager
from app.telegram_bot import TelegramBot
from app.model_trainer import ModelTrainer

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Exemple de symboles combinés pour les actions et le Forex
forex_symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
symbols = stock_symbols + forex_symbols

# Télécharger ou générer des données de marché si le fichier n'existe pas
if not os.path.exists('market_data.csv'):
    logger.info("Le fichier 'market_data.csv' n'existe pas. Téléchargement des données de marché...")
    data = yf.download(symbols, start='2022-01-01', end='2023-01-01')
    data.to_csv('market_data.csv')
    logger.info("Données de marché téléchargées et sauvegardées dans 'market_data.csv'")

# Vérifier les colonnes des données téléchargées
data = pd.read_csv('market_data.csv')
logger.info("Colonnes des données téléchargées : %s", data.columns)
numeric_data = data.select_dtypes(include=[np.number])
print("Colonnes numériques : ", numeric_data.columns)
logger.info("Colonnes numériques des données : %s", numeric_data.columns)

# Initialisation des composants
data_manager = DataManager(symbols)
trading_bot = TradingBot()
model_trainer = ModelTrainer(trading_bot)
telegram_bot = TelegramBot()

def execute_trades(trading_bot, data_manager):
    logger.info("Executing trades...")
    decisions = trading_bot.get_trading_decisions(data_manager)
    for symbol, decision_data in decisions.items():
        logger.info(f"Symbol: {symbol}, Decision: {decision_data['décision']}, "
                    f"TP: {decision_data['Take Profit']}, SL: {decision_data['Stop Loss']}")

# Planification des trades
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
    model_trainer.train_all_models(data_manager)

    logger.info("Starting reinforcement learning training...")
    trading_bot.run_reinforcement_learning('market_data.csv')

    logger.info("Starting data update and trading bot...")
    data_manager.start_data_update(interval_minutes=5)

    # Exemples d'utilisation des autres fonctionnalités
    # 1. Analyse Sentimentale
    headlines = [
        "Stock market crashes amid economic uncertainty",
        "Tech stocks rally on strong earnings reports",
        "Investors optimistic about economic recovery"
    ]
    sentiments = trading_bot.run_sentiment_analysis(headlines)
    logger.info(f"Sentiments: {sentiments}")

    # 2. Backtesting
    trading_bot.run_backtest('market_data.csv')
    
    while True:
        schedule.run_pending()
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())