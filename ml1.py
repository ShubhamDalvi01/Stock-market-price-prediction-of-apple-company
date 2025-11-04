import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import tensorflow as tf

# Fix seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

print("Script started", flush=True)

# Download historical stock data
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2025-08-24'
print(f"Downloading {ticker} data from {start_date} to {end_date}", flush=True)
data = yf.download(ticker, start=start_date, end=end_date)
print("Data downloaded", flush=True)

# Prepare data for LSTM (using Close price only)
close_prices = data['Close'].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

train_len = int(len(scaled_data)*0.8)
train_data = scaled_data[:train_len]
test_data = scaled_data[train_len-60:]

def create_sequences(dataset, step=60):
    x, y = [], []
    for i in range(step,len(dataset)):
        x.append(dataset[i-step:i,0])
        y.append(dataset[i,0])
    return np.array(x), np.array(y)

x_train_full, y_train_full = create_sequences(train_data)
x_test, y_test = create_sequences(test_data)

# Split training into train + validation sets
val_split = 0.2
val_size = int(len(x_train_full)*val_split)

x_train = x_train_full[:-val_size]
y_train = y_train_full[:-val_size]
x_val = x_train_full[-val_size:]
y_val = y_train_full[-val_size:]

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

print(f"Train samples: {x_train.shape[0]}, Val samples: {x_val.shape[0]}, Test samples: {x_test.shape[0]}", flush=True)

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(x_train.shape[1],1)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
print("Model compiled", flush=True)

# Early stopping and model checkpoint
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train model with validation and callbacks
history = model.fit(x_train, y_train, epochs=100, batch_size=32, 
                    validation_data=(x_val,y_val), 
                    callbacks=[early_stop, checkpoint], verbose=1)

print("Training completed", flush=True)

# Load best model from checkpoint
model.load_weights('best_model.h5')

# Predict test data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1,1))

rmse = np.sqrt(np.mean((predictions - actual)**2))
print(f"Test RMSE: {rmse:.4f}", flush=True)

# Prepare validation data for plotting
train = data[:train_len]
valid = data[train_len:].copy()
valid['Predictions'] = predictions.flatten()

# Predict future ~500 days iteratively
def predict_future_prices(model, last_seq, days=500):
    preds = []
    seq = last_seq.copy()
    for _ in range(days):
        p = model.predict(seq.reshape(1,60,1))[0,0]
        preds.append(p)
        seq = np.append(seq[1:], p)
    return np.array(preds)

future_days = 500
last_60 = scaled_data[-60:].flatten()
future_scaled_preds = predict_future_prices(model, last_60, future_days)
future_preds = scaler.inverse_transform(future_scaled_preds.reshape(-1,1))

# Generate future trading dates skipping weekends
last_date = data.index[-1]
future_dates = []
days_added = 0
while len(future_dates) < future_days:
    days_added += 1
    d = last_date + datetime.timedelta(days=days_added)
    if d.weekday() < 5:
        future_dates.append(d)

# Plot all results
plt.figure(figsize=(16,8))
plt.title(f'{ticker} Stock Price Prediction with LSTM')
plt.plot(data.index, data['Close'], label='Historical', color='blue')
plt.plot(valid.index, valid['Predictions'], label='Test Predictions', color='green')
plt.plot(future_dates, future_preds, label='Future Forecast', color='red', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.legend()
plt.grid(True)
plt.show()

print("Complete plotting done.", flush=True)
input("Press Enter to exit...")
