# import libraries
import math
import keras
from numpy import concatenate
from scipy.optimize import optimize
from sklearn.utils import shuffle
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# get user stock symbol
stock_symbol = input("Enter ticker:  ")

# define the start and end dates for data retrieval
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(years=10)

# retrieve the stock data using yfinance and handle missing values
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
stock_data = stock_data.dropna()
print(stock_data)

# plot 10 year price data
# plt.figure(figsize=(12,6))
# plt.plot(stock_data['Close'])
# plt.title(f"{stock_symbol} Stock Price")
# plt.xlabel("Date")
# plt.ylabel("Price [USD]")
# plt.grid(True)
# plt.show()

# normalize data using min-max scaling
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']])
normalized_df = pd.DataFrame(normalized_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

print(normalized_df)

# define input and target variables
X = normalized_df[['Open', 'High', 'Low', 'Close', 'Volume']].values
y = normalized_df['Close'].values

# split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# reshape for LSTM model
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(1, X_train.shape[2])))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# make predictions
predictions = model.predict(X_test)

# denormalize the predictions
# denormalized_predictions = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), predictions), axis=1))[:, -1].reshape(-1, 1)
min_close = stock_data['Close'].min()
max_close = stock_data['Close'].max()

denormalized_predictions = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), predictions), axis=1))[:, -1].reshape( -1, 1)

denormalized_predictions = (denormalized_predictions * (max_close - min_close)) + min_close

# plot the predicted and actual prices
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index[-len(denormalized_predictions):], denormalized_predictions, label='Predicted')
plt.plot(stock_data.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual')
plt.title(f"{stock_symbol} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price [USD]")
plt.legend()
plt.grid(True)
plt.show()

