# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from datetime import timedelta
import yfinance as yf

# Download the stock data
df = yf.download('ICICIBANK.NS', start="2010-01-01", end="2024-11-10").reset_index()

# Calculate technical indicators
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
df['Returns'] = df['Close'].pct_change()

# Drop NaNs after calculating indicators
df.dropna(inplace=True)

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close', 'MA10', 'MA50', 'MA200', 'Returns']])

# Define the sequence length (updated to 50 for past 50 days input)
TIME_STEPS = 50

# Create sequences
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i])
        y.append(data[i, 0])  # Predicting 'Close' price
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, TIME_STEPS)

# Split into training and validation sets
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Build the LSTM model with added complexity
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='huber')

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# Prediction function for the next 60 days with dimensionality matching
def predict_next_n_days(model, last_sequence, n_days, scaler):
    """
    Predicts the next n_days closing prices based on the last 50 days of scaled data.
    
    Parameters:
    - model: Trained LSTM model.
    - last_sequence: Last sequence of scaled data used as input to predict future values.
    - n_days: Number of days to predict.
    - scaler: Fitted MinMaxScaler object.
    
    Returns:
    - predictions: list of predicted prices in original scale.
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_days):
        # Reshape current sequence to fit the model input
        current_sequence_reshaped = current_sequence.reshape((1, TIME_STEPS, current_sequence.shape[1]))

        # Predict the next day
        next_day_scaled = model.predict(current_sequence_reshaped, verbose=0)
        
        # Inverse transform to get the actual price and add to predictions list
        next_day_price = scaler.inverse_transform(np.array([[next_day_scaled[0][0], 0, 0, 0, 0]]))[0][0]
        predictions.append(next_day_price)
        
        # Update the sequence with the predicted value
        # Fill other features with the last observed values in the sequence
        next_day_full = np.zeros(current_sequence.shape[1])
        next_day_full[0] = next_day_scaled  # Update the 'Close' price
        next_day_full[1:] = current_sequence[-1, 1:]  # Copy remaining features

        # Append to the sequence and drop the first step to maintain the window length
        current_sequence = np.append(current_sequence[1:], [next_day_full], axis=0)
    
    return predictions

# Get the last sequence for prediction
last_sequence = X_val[-1]

# Predict the next 60 days
future_predictions = predict_next_n_days(model, last_sequence, 60, scaler)

# Convert predictions to dates for plotting
last_date = df['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 61)]

# Plot results
plt.figure(figsize=(14, 7))

# Plot last 50 days of actual data
plt.plot(df['Date'].iloc[-100:], df['Close'].iloc[-100:], label="Actual Prices", marker="o")

# Plot predicted values
plt.plot(future_dates, future_predictions, label="Predicted Prices", marker="o", linestyle="--", color="red")
plt.title("WIPRO Stock Price Prediction: Last 50 Days and 60-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Price (₹)")
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()