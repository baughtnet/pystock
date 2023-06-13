
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Get stock ticker from user
stock_symbol = input("Enter stock ticker: ")

# Set the start and end dates for the data retrieval
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(years=10)

# Download stock data from Yahoo Finance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
stock_data = stock_data.dropna()

# Extract the closing prices
close_prices = np.array(stock_data['Close']).reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(close_prices)

# Time horizon for prediction
num_days = 22 # 1 month prediction

# Prepare the data for LSTM
x, y = [], []
for i in range(num_days, len(normalized_data)):
    x.append(normalized_data[i - num_days:i, 0])
    y.append(normalized_data[i, 0])
x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=100, batch_size=32)

# Make predictions for the future
future_data = normalized_data[-num_days:]
future_data = np.reshape(future_data, (1, future_data.shape[0], 1))
predicted_data = model.predict(future_data)
predicted_data = scaler.inverse_transform(predicted_data)

# Generate the graph
plt.figure()
plt.plot(stock_data.index, stock_data['Close'], linestyle=':', label='Historical Data')
plt.plot([stock_data.index[-1] + pd.DateOffset(days=1), stock_data.index[-1] + pd.DateOffset(days=1) + pd.DateOffset(months=1)], [stock_data['Close'][-1], predicted_data[0][0]], label='Predicted Data')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title(f'Future Prediction for {stock_symbol}')
plt.legend()
plt.xticks(rotation=45)
plt.show()
