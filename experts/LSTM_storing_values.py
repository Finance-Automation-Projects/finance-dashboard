import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from datetime import timedelta

def train_and_forecast(stocks: list[str], forecast_days: int = 7) -> pd.DataFrame:
    """
    Train the LSTM model on individual stocks and forecast their prices for the next 7 days.
    
    Args:
        stocks (list[str]): List of stock tickers to train and forecast.
        forecast_days (int): Number of days to forecast. Default is 7.
        
    Returns:
        pd.DataFrame: DataFrame containing the forecasted prices for each stock.
    """
    def create_model(input_shape):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(units=50)))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def predict_next_n_days(model, last_sequence, n_days, scaler):
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_days):
            current_sequence_reshaped = current_sequence.reshape((1, current_sequence.shape[0], current_sequence.shape[1]))
            next_day_scaled = model.predict(current_sequence_reshaped, verbose=0)
            next_day_price = scaler.inverse_transform(np.array([[next_day_scaled[0][0], 0, 0, 0, 0]]))[0][0]
            predictions.append(next_day_price)
            next_day_full = np.zeros(current_sequence.shape[1])
            next_day_full[0] = next_day_scaled
            next_day_full[1:] = current_sequence[-1, 1:]
            current_sequence = np.append(current_sequence[1:], [next_day_full], axis=0)
        
        return predictions

    forecast_table = pd.DataFrame()

    for stock in stocks:
        # Download the stock data
        df = yf.download(stock, start="2010-01-01", end="2024-11-10").reset_index()
        
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
        
        # Define the sequence length
        TIME_STEPS = 50
        
        # Create sequences and labels
        X = []
        y = []
        for i in range(TIME_STEPS, len(scaled_data)):
            X.append(scaled_data[i-TIME_STEPS:i])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split the data into training and validation sets
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Create and train the model
        model = create_model((X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
        
        # Get the last sequence for prediction
        last_sequence = X_val[-1]
        
        # Predict the next 7 days
        future_predictions = predict_next_n_days(model, last_sequence, forecast_days, scaler)
        
        # Convert predictions to dates for the table
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        
        # Create a DataFrame for the predictions
        stock_forecast = pd.DataFrame({'Date': future_dates, 'Forecast': future_predictions})
        stock_forecast['Stock'] = stock
        
        # Append to the forecast table
        forecast_table = pd.concat([forecast_table, stock_forecast], ignore_index=True)
    
    return forecast_table

# Example usage
stocks = ['ICICIBANK.NS', 'TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS']  # Add more stock tickers as needed
forecast_table = train_and_forecast(stocks)
print(forecast_table)