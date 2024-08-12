from flask import Blueprint, jsonify
from .trading import TradingBot, DataManager, RealTimeTrainer
import schedule
import time

main_blueprint = Blueprint('main', __name__)

symbols = ['AAPL', 'GOOGL', 'MSFT']  # Exemple de symboles
data_manager = DataManager(symbols)
trading_bot = TradingBot()
trainer = RealTimeTrainer(data_manager, trading_bot)

data_manager.start_data_update(interval_minutes=5)
trainer.start_training(interval_minutes=10)

@main_blueprint.route('/status', methods=['GET'])
def status():
    decisions = trading_bot.get_trading_decisions(data_manager)
    return jsonify({
        "status": "Modèle en cours d'exécution",
        "décisions": decisions
    })

@main_blueprint.route('/server-stats', methods=['GET'])
def server_status():
    stats = trainer.get_server_stats()
    return jsonify(stats)