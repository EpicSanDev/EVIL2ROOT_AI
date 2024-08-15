---

# **AI-Powered Trading Bot**

Welcome to the **AI-Powered Trading Bot** repository! This project is a comprehensive, high-performance trading bot that integrates machine learning, reinforcement learning, and sentiment analysis to make informed trading decisions across multiple markets, including stocks, forex, and cryptocurrencies.

## 🌟 **Features**

### 🚀 **Multi-Asset Support**
- **Stocks, Forex, and Cryptocurrencies**: Supports a vast range of assets, including major stock symbols, forex pairs, and popular cryptocurrencies.

### 🧠 **Advanced AI Models**
- **Price Prediction**: Uses sophisticated machine learning models to predict future prices based on historical data.
- **Risk Management**: Implements AI-driven risk assessment to avoid trades with unfavorable risk/reward ratios.
- **Take Profit and Stop Loss**: Automatically calculates optimal TP and SL levels to maximize profits and minimize losses.
- **Technical Indicators**: Incorporates technical analysis through various indicators, including moving averages and RSI, to refine trading decisions.

### 🎯 **Reinforcement Learning**
- **Adaptive Learning**: Trains a reinforcement learning agent to continuously improve trading strategies based on past performance.
- **Real-Time Decision Making**: Executes trades based on real-time market data and dynamically adjusts strategies as markets evolve.

### 💬 **Sentiment Analysis**
- **Market Sentiment**: Analyzes news headlines and other text-based data to gauge market sentiment, which influences trading decisions.
- **Natural Language Processing**: Utilizes NLP techniques to derive sentiment scores from various sources, providing an additional layer of decision-making insight.

### 📈 **Backtesting**
- **Historical Data Testing**: Run simulations on historical market data to evaluate the performance of trading strategies before deploying them in live markets.
- **Comprehensive Analysis**: Provides detailed reports on strategy performance, including profit/loss, drawdowns, and other key metrics.

### 🕒 **Real-Time Market Scanning**
- **Continuous Market Monitoring**: Scans multiple markets in real-time to identify trading opportunities as they arise.
- **Automated Trading Execution**: Executes trades automatically based on pre-defined strategies and market conditions.

### 📊 **Web Dashboard**
- **Performance Monitoring**: View live updates on the bot’s performance, including CPU/GPU usage, trade execution, and financial metrics.
- **Interactive Charts**: Use Plotly-powered charts to visualize market data, trade history, and more.
- **User-Friendly Interface**: A clean and intuitive web interface for monitoring and controlling the trading bot.

### 🔔 **Telegram Notifications**
- **Real-Time Alerts**: Receive instant notifications about trading activity, model training updates, and system status directly to your Telegram account.
- **Customizable Messages**: Stay informed about important events without needing to be logged into the dashboard.

## 📚 **Getting Started**

### Prerequisites
- **Python 3.8+**
- **PyTorch** (for deep learning models)
- **Stable Baselines3** (for reinforcement learning)
- **Flask** (for web interface)
- **YFinance** (for market data)
- **Plotly** (for charting)
- **Telegram API** (for notifications)

### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/EpicSanDev/EVIL2ROOT_AI.git
pip install -r requirements.txt
```

### Configuration
Set up your API keys and other configuration details in the `config.py` file:
```python
TELEGRAM_API_KEY = 'your-telegram-api-key'
```

### Running the Bot
To start the trading bot with the web interface:
```bash
python run.py
```

## 🤝 **Contributing**
Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

## 📄 **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🌐 **Connect**
For questions, feedback, or just to connect, reach out via [Telegram](https://t.me/bastienjavaux).

---
