import os
import logging
import asyncio
import yfinance as yf
from app.trading import TradingBot, DataManager
from app.telegram_bot import TelegramBot
from app.model_trainer import ModelTrainer
import schedule
import pandas as pd
import numpy as np
from app import create_app



file_handler = logging.FileHandler('trading_bot.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Starting the trading bot...")
# Configuration de la journalisation
logging.basicConfig(
    level=logging.INFO,  # Niveau de log
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format des messages de log
    handlers=[
        logging.FileHandler("trading_bot.log"),  # Enregistre les logs dans un fichier
        logging.StreamHandler()  # Affiche les logs dans la console
    ]
)


logger = logging.getLogger(__name__)


forex_symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "EURGBP=X", "EURJPY=X", "EURCHF=X", "GBPJPY=X", "AUDJPY=X", "NZDUSD=X", "USDCHF=X", "EURNZD=X", "GBPAUD=X", "GBPCAD=X", "NZDJPY=X", "CADJPY=X", "AUDNZD=X", "USDHKD=X", "USDSGD=X", "EURCAD=X", "GBPCHF=X", "AUDCAD=X", "AUDCHF=X", "EURHKD=X"]

stock_symbols = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX", "DIS", "BABA", 
    "INTC", "AMD", "CSCO", "ORCL", "IBM", "ADBE", "CRM", "PYPL", "V", "MA", 
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BK", "BLK", "AXP", "COF", 
    "T", "VZ", "TMUS", "CMCSA", "CHTR", "TMO", "UNH", "JNJ", "PFE", "MRK", 
    "ABT", "ABBV", "LLY", "BMY", "AMGN", "GILD", "CVS", "WBA", "CI", 
    "XOM", "CVX", "COP", "PSX", "VLO", "BP", "RDS.B", "ENB", 
    "TSM", "ASML", "QCOM", "AVGO", "TXN", "AMAT", "LRCX", "KLAC", "MU", "NXPI", 
    "SHOP", "SQ", "SNAP", "PINS", "ZM", "DOCU", "ROKU", "SPOT", "PLTR", 
    "BA", "LMT", "GD", "NOC", "TXT", "HII", "LHX", "TDG", "HEI", 
    "CAT", "DE", "CARR", "OTIS", "IR", "DHR", "MMM", "HON", "GE", "BA", 
    "PEP", "KO", "MDLZ", "KHC", "GIS", "K", "COST", "WMT", "TGT", "DG", 
    "NKE", "LULU", "VFC", "RL", "GPS", "ANF", "AEO", "UA", "SKX", "CROX", 
    "HD", "LOW", "BBY", "WBA", "CVS", "GME", "SIG", "NWSA", 
    "FDX", "UPS", "LSTR", "CHRW", "JBHT", "EXPD", "XPO", "KNX", "SNDR", "WERN"
]

crypto_symbols = [
    "BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "ADA-USD", "BCH-USD", "LINK-USD", "DOT-USD", "XLM-USD", "BNB-USD",
    "USDT-USD", "EOS-USD", "TRX-USD", "NEO-USD", "ETC-USD", "VET-USD", "ZRX-USD", "BAT-USD", "MKR-USD", "ZIL-USD",
    "XTZ-USD", "DASH-USD", "XMR-USD", "DOGE-USD", "OMG-USD", "DCR-USD", "QTUM-USD", "LSK-USD", "WAVES-USD", "KMD-USD",
    "ZEC-USD", "LRC-USD", "NANO-USD", "ICX-USD", "IOST-USD", "MANA-USD", "ENJ-USD", "SNT-USD", "GNT-USD", "REP-USD",
    "CVC-USD", "STORJ-USD", "ANT-USD", "FUN-USD", "MITH-USD", "GAS-USD", "AION-USD", "ONT-USD", "ZEN-USD", "STEEM-USD"
]

# Combine all symbols
symbols = stock_symbols + forex_symbols + crypto_symbols

additional_symbols = [
    "F", "GM", "TM", "HMC", "NSANY", "VWAGY", "BMWYY",
    "HYMTF", "RACE", "TSLA", "NIO", "XPEV", "LI", "BYDDF", "FUV", "NKLA",
    "UBER", "LYFT", "GRAB", "SQM", "ALB", "PLL", "LAC", "MP", "AVNT",
    "GLNCY", "BHP", "RIO", "VALE", "FCX", "SCCO", "AA", "CENX", "HAYN",
    "ATI", "CMC", "MT", "STLD", "X", "CLF", "NUE", "RS", "TX",
    "CRS", "RYI", "TSN", "HRL", "PPC", "BWXT", "TDY",
    "J", "FLR", "KBR", "PWR", "MYRG", "MTZ", "PRIM", "AGX", "GVA", "DY",
    "SSP", "GCI", "TGNA", "NXST", "SBGI", "GTN", "MEG", "SNI", "DISCA",
    "QRTEA", "DISCK", "FWONK", "BIDU", "PDD", "JD", "TCEHY", "NTES", "BILI",
    "IQ", "WB", "MOMO", "HUYA", "DOYU", "YY", "ATHM", "BZUN", "TAL", "EDU",
    "HRHTF", "AXAHY", "ALL", "AIG", "CB", "TRV", "PGR", "CINF", "MKL", "RE",
    "HIG", "PRU", "MET", "LNC", "UNM", "AFL", "AIZ", "AFG", "L", "CNO",
    "GL", "MFC", "SLF", "PNX", "THG", "RLI", "SIGI", "WRB", "CNA", "MCY"
]

symbols = symbols + additional_symbols

# Télécharger ou générer des données de marché si le fichier n'existe pas
if not os.path.exists('market_data.csv'):
    logger.info("Le fichier 'market_data.csv' n'existe pas. Téléchargement des données de marché...")
    data = yf.download(symbols, start='2001-01-01', end='2024-01-01')
    data.to_csv('market_data.csv')
    logger.info("Données de marché téléchargées et sauvegardées dans 'market_data.csv'")

# Vérifier les colonnes des données téléchargées
data = pd.read_csv('market_data.csv')
logger.info("Colonnes des données téléchargées : %s", data.columns)
numeric_data = data.select_dtypes(include=[np.number])
logger.info("Colonnes numériques des données : %s", numeric_data.columns)
from data_cleaner import clean_data


# Initialisation des composants
data_manager = DataManager(symbols)
trading_bot = TradingBot()
model_trainer = ModelTrainer(trading_bot)
telegram_bot = TelegramBot()

# Paths
input_file = 'market_data.csv'
cleaned_file = 'market_data_cleaned_auto.csv'

# Clean the data
clean_data(input_file, cleaned_file)


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
    trading_bot.run_reinforcement_learning('market_data_cleaned_auto.csv')

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
    trading_bot.run_backtest('market_data_cleaned_auto.csv')
    
    while True:
        schedule.run_pending()
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
