from flask import Blueprint, jsonify
from .trading import TradingBot
from .utils import server_stats

main_blueprint = Blueprint('main', __name__)

@main_blueprint.route('/status', methods=['GET'])
def status():
    bot = TradingBot()
    decision, indicator = bot.get_trading_decision()
    return jsonify({"status": "Modèle en cours d'exécution", "décision": decision, "indicateur ajusté": indicator})

@main_blueprint.route('/server-stats', methods=['GET'])
def server_status():
    stats = server_stats()
    return jsonify(stats)