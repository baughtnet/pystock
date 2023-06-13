import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics
from sklearn.preprocessing import MinMaxScaler

stock_symbol = input("Enter ticker:  ")

end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(years=10)

stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
stock_data = stock_data.dropna()

df = stock_data

L=len(df)
print(L)

high = np.array([df.iloc[:,2]])
low = np.array([df.iloc[:,3]])
close = np.array([df.iloc[:,4]])

plt.figure(1)
h, = plt.plot(high[0,:])
l, = plt.plot(low[0,:])
c, = plt.plot(close[0,:])

plt.legend([h,l,c],["High","Low","Close"])
plt.show(block=False)

#print("made the graph!")

x = np.concatenate([high,low],axis= 0)
x = np.transpose(x)

y = close
y = np.transpose(y)

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

scaler1 = MinMaxScaler()
scaler1.fit(y)
y = scaler1.transform(y)

x = x[:-1] # Remove the last data point from x
y = y[1:] # Shift y by one position to the left
subset = int(len(y) * 0.8)
x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
print(x.shape)

model = Sequential()
model.add(LSTM(100,activation='tanh',input_shape=(1,2),recurrent_activation= 'hard_sigmoid'))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics = [metrics.mae])
model.fit(x,y,epochs=10,batch_size= 1,verbose= 2)

predict = model.predict(x,verbose=1)
print(predict)

plt.figure(2)
plt.scatter(y,predict)
plt.show(block = False)


plt.figure(3)
test, = plt.plot(y[:subset])
prediction, = plt.plot(predict[:subset])
plt.legend([prediction, test], ["Predicted Data", "Real Data"])
plt.show(block = False)

subset = int(len(y) * 0.8)
subset_3weeks = int(len(y) * 0.8) - 15  # Adjust the number of weeks as per your requirement

predict = np.squeeze(predict)  # Convert predict to a 1D array
plt.figure(4)
test, = plt.plot(range(subset_3weeks, subset_3weeks + len(y[-subset_3weeks:-subset])), y[-subset_3weeks:-subset])
prediction, = plt.plot(range(subset_3weeks, subset_3weeks + len(predict[-subset_3weeks:-subset])), predict[-subset_3weeks:-subset])
future_prediction, = plt.plot(range(subset_3weeks + len(y[-subset_3weeks:-subset]), subset_3weeks + len(y[-subset_3weeks:])), predict[-subset_3weeks:])
plt.legend([prediction, test, future_prediction], ["Predicted Data", "Real Data", "Future Prediction"])
plt.show()
