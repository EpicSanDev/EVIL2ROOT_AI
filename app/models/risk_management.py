from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

class RiskManagementModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def train(self, data, symbol):
        self.scalers[symbol] = StandardScaler()
        features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        features_scaled = self.scalers[symbol].fit_transform(features)
        target = data['Close'].pct_change().fillna(0).values  # Return as a proxy for risk

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features_scaled, target)
        self.models[symbol] = model

    def predict(self, data, symbol):
        features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        features_scaled = self.scalers[symbol].transform(features)
        predicted_risk = self.models[symbol].predict(features_scaled[-1].reshape(1, -1))
        return predicted_risk[0]