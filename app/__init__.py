from flask import Flask
from app.trading import TradingBot, DataManager
trading_bot = TradingBot()
data_manager = DataManager
def create_app():
    app = Flask(__name__)

    # Import and register the main blueprint
    from .routes import main_blueprint
    app.register_blueprint(main_blueprint)

    return app