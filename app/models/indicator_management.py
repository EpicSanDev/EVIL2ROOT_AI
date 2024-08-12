import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

class IndicatorManagementModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def train(self, data, symbol):
        indicators = self.calculate_indicators(data)
        self.scalers[symbol] = MinMaxScaler()
        indicators_scaled = self.scalers[symbol].fit_transform(indicators)

        model = Ridge(alpha=1.0)
        model.fit(indicators_scaled, data['Close'].values)
        self.models[symbol] = model

    def calculate_indicators(self, data):
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        indicators = data[['SMA_20', 'SMA_50']].fillna(0).values
        return indicators

    def predict(self, data, symbol):
        indicators = self.calculate_indicators(data)
        indicators_scaled = self.scalers[symbol].transform(indicators)
        predicted_indicator = self.models[symbol].predict(indicators_scaled[-1].reshape(1, -1))
        return predicted_indicator[0]