import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Télécharger les ressources nltk si nécessaire
import nltk
nltk.download('punkt')

# Prétraitement des données
def preprocess_text(texts, max_len=100, max_words=10000):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer

# Exemple d'ensemble de données pour l'entraînement
# Supposons que vous avez un DataFrame avec deux colonnes: 'text' et 'label'
# text: les textes (titres, tweets, etc.)
# label: 0 pour négatif, 1 pour positif
# dataset = pd.read_csv('path_to_your_dataset.csv')

# texts = dataset['text'].values
# labels = dataset['label'].values

# Pour l'exemple, utilisons des données fictives
texts = ["The stock market is going up", "The economy is crashing", "Great earnings reports from tech companies"]
labels = [1, 0, 1]

# Prétraitement
padded_sequences, tokenizer = preprocess_text(texts)

# Diviser les données en ensemble d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Définir le modèle LSTM
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, np.array(y_train), validation_data=(X_val, np.array(y_val)), epochs=10, batch_size=32)

# Enregistrer le modèle
model.save("sentiment_model.h5")

# Fonction pour prédire le sentiment
def predict_sentiment(text, model, tokenizer, max_len=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    return prediction[0][0]
def sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

def analyze_headlines(headlines):
    sentiments = [sentiment_analysis(headline) for headline in headlines]
    return sentiments
# Exemple d'utilisation
sentiment_score = predict_sentiment("The stock market is on the rise", model, tokenizer)
print(f"Sentiment score: {sentiment_score}")