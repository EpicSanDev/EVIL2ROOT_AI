import psutil
import yfinance as yf

def server_stats():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    return {"cpu_usage": cpu_usage, "memory_usage": memory_usage}