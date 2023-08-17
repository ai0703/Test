pip install streamlit
import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import ta
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import joblib
import matplotlib.pyplot as plt

st.title('Stock Price Prediction App')

# Collect Data from Yahoo Finance
ticker = 'TSLA'
start_date = '2012-01-01'
end_date = '2022-12-31'
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate technical indicators
data['MA'] = ta.trend.SMAIndicator(data['Close'], window=10).sma_indicator()
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

# ... (Rest of your code)

# Display the ensemble accuracy using a bar plot
st.bar_chart(pd.DataFrame({'Model': model_names, 'Accuracy': accuracy_values}).set_index('Model'))

# ... (Rest of your code)

# User input for custom date range
custom_start_date = st.text_input("Enter the custom start date (YYYY-MM-DD):")
custom_end_date = st.text_input("Enter the custom end date (YYYY-MM-DD):")

# ... (Rest of your code)

# Display ensemble predictions
st.write("Ensemble Predictions for Custom Data:")
st.write(ensemble_predictions_custom)

# Visualize ensemble predictions
st.write("Ensemble Predictions Visualization:")
st.line_chart(pd.DataFrame({'Date': dates_custom, 'Ensemble Prediction': ensemble_predictions_custom}).set_index('Date'))

# ... (Rest of your code)

# Calculate and display accuracy for each model and ensemble manually
st.write("Model Accuracies:")
st.write(f"Naive Bayes Accuracy: {naive_bayes_accuracy:.4f}")
st.write(f"CNN Accuracy: {cnn_accuracy:.4f}")
st.write(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")

# Display predictions
st.write("Predictions for Custom Data:")
st.write("Naive Bayes Predictions:")
st.write(naive_bayes_predictions_custom)

st.write("CNN Predictions:")
st.write(cnn_predictions_custom)
