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
def train(self, data, symbol):
    self.scalers[symbol] = MinMaxScaler(feature_range=(0, 1))
    scaled_data = self.scalers[symbol].fit_transform(data['Close'].values.reshape(-1,1))
    X_train, y_train = [], []

    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = self.build_model()
    model.fit(X_train, y_train, batch_size=32, epochs=50)
    self.models[symbol] = model