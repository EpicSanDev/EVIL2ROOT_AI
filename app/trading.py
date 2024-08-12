from .utils import get_stock_data
from .models.price_prediction import PricePredictionModel
from .models.risk_management import RiskManagementModel
from .models.indicator_management import IndicatorManagementModel
from .utils import get_stock_data

class TradingBot:
    def __init__(self):
        self.price_model = PricePredictionModel()
        self.risk_model = RiskManagementModel()
        self.indicator_model = IndicatorManagementModel()

    def train_models(self):
        data = get_stock_data("AAPL")
        self.price_model.train(data)
        self.risk_model.train(data)
        self.indicator_model.train(data)

    def get_trading_decision(self):
        data = get_stock_data("AAPL")
        predicted_price = self.price_model.predict(data)
        predicted_risk = self.risk_model.predict(data)
        adjusted_indicator = self.indicator_model.predict(data)

        # Exemple de décision basée sur les modèles
        if predicted_risk < 0.02:  # Exemple de seuil de risque
            if predicted_price > data['Close'][-1]:
                return "Acheter", adjusted_indicator
            else:
                return "Vendre", adjusted_indicator
        return "Ne rien faire", adjusted_indicator

    def get_prediction(self):
        data = get_stock_data("AAPL")
        prediction = self.model.predict(data)
        return prediction