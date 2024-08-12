# app/models/sentiment_analysis.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('punkt')

def sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

def analyze_headlines(headlines):
    sentiments = [sentiment_analysis(headline) for headline in headlines]
    return sentiments