# this program is meant to take from a few previous programs and implement 
# functions in order to do more complex tasks later on

# import libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def get_data():
    ticker = input("Enter stock ticker:  ")

    prediction_period = input("Enter Prediction Length(1 Day, 1 Week, 1 Month, 6 Month, 1 Year, 3 Year or All)")

    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=10)

    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data.dropna()

    return stock_data, prediction_period

def build_LSTM(stock_data, prediction_period):
    print(stock_data)
    print(prediction_period)

get_data()
build_LSTM(stock_data, prediction_period)


