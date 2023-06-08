# import libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# get user stock symbol
stock_symbol = input("Enter ticker:  ")

# define the start and end dates for data retrieval
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(years=10)

# retrieve the stock data using yfinance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

print(stock_data)

plt.figure(figsize=(12,6))
plt.plot(stock_data['Close'])
plt.title(f"{stock_symbol} Stock Price")
plt.xlabel("Date")
plt.ylabel("Price [USD]")
plt.grid(True)
plt.show()
