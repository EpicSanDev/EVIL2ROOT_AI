from flask import Blueprint, render_template, jsonify
import psutil
import plotly.graph_objs as go
import psutil
import plotly.io as pio
from app.trading import TradingBot
from app.trading import DataManager
from app.model_trainer import ModelTrainer

main_blueprint = Blueprint('main', __name__)

# Initialize necessary components
symbols = ["AAPL", "GOOGL", "BTC-USD"]  # Example symbols
data_manager = DataManager(symbols)
trading_bot = TradingBot()
model_trainer = ModelTrainer(trading_bot)

@main_blueprint.route('/')
def dashboard():
    # CPU/GPU Performance Data
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    # Prepare CPU usage chart
    cpu_chart = go.Figure(data=[go.Bar(x=['CPU Usage'], y=[cpu_percent], marker=dict(color='rgb(55, 83, 109)'))])
    cpu_chart.update_layout(title='CPU Usage (%)')

    # Convert the chart to JSON
    cpu_chart_json = pio.to_json(cpu_chart)

    return render_template('dashboard.html', 
                           cpu_percent=cpu_percent,
                           memory_info=memory_info,
                           cpu_chart=cpu_chart_json)
@main_blueprint.route('/bot_status')
def bot_status():
    status = {
        'state': 'running',  # Update this according to your bot's real state
        'signals': trading_bot.get_latest_signals()
    }
    return jsonify(status)