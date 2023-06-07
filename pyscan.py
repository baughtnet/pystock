# import libraries
from pandas_datareader import data as pdr
from sqlalchemy.sql import False_
from yahoo_fin import stock_info as si
from pandas import ExcelWriter
import yfinance as yf
import pandas as pd
import datetime
import time

yf.pdr_override
# variables
tickers = si.tickers_sp500()
tickers = [item.replace(".", "-") for item in tickers]
index_name = '^GSPC'
start_date = datetime.datetime.now() - datetime.timedelta(days=365)
end_date = datetime.date.today()
exportList = pd.DataFrame(columns=['Stock', "RS_Rating", "50 Day MA", "150 Day MA", "200 Day MA", "52 Week Low", "52 Week High"])
return_multiples = []

# index returns
index_df = pdr.get_data_yahoo(index_name, start_date, end_date)
index_df['Percent Change'] = index_df['Adj Close'].pct_change()
index_return = (index_df['Percent Change'] + 1).cumprod()[-1]

# find top 30% performig stocks relative to the S&P 500
for ticker in tickers:
    #download historical data as csv for each stock(makes the process faster)
    df = pdr.get_data_yahoo(ticker, start_date, end_date)
    df.to_csv(f'{ticker}.csv')

    #calculating returns relative to the market(returns multiple)
    df['Percent Change'] = df['Adj Close'].pct_change()
    stock_return = (df['Percent Change'] + 1).cumprod()[-1]

    returns_multiple = round((stock_return / index_return), 2)
    returns_multiples.extend([returns_multiple])

    print(f'Ticker: {ticker}; Returns Multiple against S&P 500: {returns_multiple}\n')
    time.sleep(1)

# creating dataframe of only top 30%
rs_df = pd.DataFrame(list(zip(tickers, returns_multiples)), columns=['Ticker', 'Returns_multiple'])
rs_df['RS_Rating'] = rs_df.Returns_multiple.rank(pct=True) * 100
rs_df = rs_df[rs_df.RS_Rating >= rs_df.RS_Rating.quantile(.70)]

# checking minervini conditions of top 30% of stocks in given list
rs_stocks = rs_df['Ticker']
for stock in rs_stocks:
    try:
        df = pd.read_csv(f'{stock}.csv', index_col=0)
        sma = [50, 150, 200]
        for x in sma:
            df["SMA_"+str(x)] = round(df['Adj Close'].rolling(window=x).mean(), 2)

        # storing required values
        currentClose = df["Adj Close"][-1]
        moving_average_50 = df["SMA_50"][-1]
        moving_average_150 = df["SMA_150"][-1]
        moving_average_200 = df["SMA_200"][-1]
        low_of_52week = round(min(df["Low"][-260]), 2)
        high_of_52week = round(min(df["High"][-260]), 2)
        RS_Rating = round(rs_df[rs_df['Ticker']==stock].RS_Rating.tolist()[0])

        try:
            moving_average_200_20 = df["SMA_200"][-20]
        except Exception:
            moving_average_200_20 = 0

        # condition 1 --> Current Price > 150 SMA and > 200 SMA
        condition_1 = currentClose > moving_average_150 > moving_average_200

        # condition 2 --> 150 SMA and > 200 SMA
        condition_2 = moving_average_150 > moving_average_200
        
        # condition 3 --> 200 SMA trending up for at least 1 month
        condition_3 = moving_average_200 > moving_average_200_20

        # condition 4 --> 50 SMA > 150 SMA and 50 SMA > 200 SMA
        condition_4 = moving_average_50 > moving_average_150 > moving_average_200

        # condition 5 --> Current price > 50 SMA
        condition_5 = currentClose > moving_average_50

        # condition 6 --> current price is at least 30% above 52 week low
        condition_6 = currentClose >= (1.3*low_of_52week)

        # condition 7 --> current price is within 25% of 52 week high
        condition_7 = currentClose >= (.75*high_of_52week)

        # if all conditions above are true, add stock to exportList
        if(condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6 and condition_7):
            exportList = exportList.append({'Stock': stock, "RS_Rating": RS_Rating, "50 Day MA": moving_average_50, "150 Day MA": moving_average_150, "200 Day MA": moving_average_200, "52 Week Low": low_of_52week, "52 Week High": high_of_52week}, ignore_index=True)
            print(stock + " made the Minervini Requirements")
    except Exception as e:
        print(e)
        print(f"Could not gather data on {stock}")

exportList = exportList.sort_values(by='RS_Rating', ascending=False)
print('\n', exportList)
writer = ExcelWriter("ScreenOutput.xlsx")
exportList.to_excel(writer, "Sheet 1")
writer.save()

