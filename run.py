from app.trading import TradingBot, DataManager
import schedule

# Exemple de symboles combinés pour les actions et le Forex
forex_symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA"]
symbols = stock_symbols + forex_symbols

data_manager = DataManager(symbols)
trading_bot = TradingBot()

def execute_trades(trading_bot, data_manager):
    decisions = trading_bot.get_trading_decisions(data_manager)
    for symbol, decision_data in decisions.items():
        print(f"Symbol: {symbol}, Decision: {decision_data['décision']}, "
              f"TP: {decision_data['Take Profit']}, SL: {decision_data['Stop Loss']}")

schedule.every(1).minutes.do(execute_trades, trading_bot, data_manager)

if __name__ == "__main__":
    data_manager.start_data_update(interval_minutes=5)
    while True:
        schedule.run_pending()
        time.sleep(1)