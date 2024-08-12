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