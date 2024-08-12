from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class TpSlManagementModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(2))  # Deux sorties : TP et SL
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, data, symbol):
        self.scalers[symbol] = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scalers[symbol].fit_transform(data['Close'].values.reshape(-1,1))
        X_train, y_train = [], []

        for i in range(60, len(scaled_data) - 2):  # Assurez-vous qu'il reste assez de données pour TP et SL
            X_train.append(scaled_data[i-60:i, 0])
            # Exemple simple : TP = prix de fermeture à t+1, SL = prix de fermeture à t+2
            tp = scaled_data[i+1, 0]
            sl = scaled_data[i+2, 0]
            y_train.append([tp, sl])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = self.build_model()
        model.fit(X_train, y_train, batch_size=32, epochs=50)
        self.models[symbol] = model

    def predict(self, data, symbol):
        last_60_days = data['Close'][-60:].values
        last_60_days_scaled = self.scalers[symbol].transform(last_60_days.reshape(-1, 1))
        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_tp_sl = self.models[symbol].predict(X_test)
        predicted_tp_sl = self.scalers[symbol].inverse_transform(predicted_tp_sl)
        return predicted_tp_sl[0][0], predicted_tp_sl[0][1]  # Retourne TP et SL