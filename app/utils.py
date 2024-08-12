import yfinance as yf
import psutil

def get_stock_data(ticker):
    data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
    return data

def server_stats():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    return {"cpu_usage": cpu_usage, "memory_usage": memory_usage}