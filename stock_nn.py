#利用yahoo蒐集股市資料

# pip install yfinance
import numpy as np
from pandas_datareader import data as pdr
import pandas as pd
import yfinance as yf
yf.pdr_override()

# 取得股市各股資料
data = pdr.get_data_yahoo("2330.TW")

for i in range(len(data)):
    if (data.Volume[i] == '0'):
        data = data.drop(i)

# 繪製各股股價歷史行情
import matplotlib.pyplot as plt
data["Close"].plot()

# 股市資料預處理
days = 400
df = pd.DataFrame()
df["volume"] = data["Volume"][-days:]   #可訂定幾筆資料，例如[-720:], 今天往前取720天
df["open"] = data["Open"][-days:]
df["high"] = data["High"][-days:]
df["low"] = data["Low"][-days:]
df["close"] = data["Close"][-days:]

#資料存取
df.to_csv("stock_origin.csv")
data = pd.read_csv("stock_origin.csv")

#股市交易資料日期欄位的處理
data = data.drop(data.columns[0], axis=1)

#資料的正規化
def normalize(data):
    data_norm = data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return data_norm
dataset_norm = normalize(data)
dataset_norm

#監督式學習的股市資料整理
pday = 1  ##假設預測明天股市收盤價
data_tmp = dataset_norm.copy()

data_tmp2 = data_tmp['close'].iloc[pday:]
data_tmp2 = data_tmp2.reset_index(drop=True)
data_tmp2.name = 'close after ' + str(pday) + ' days'
data_tmp2

data_pre = data_tmp[-pday:]  #最後一天的資料保留，作為預測明天
data_tmp1 = data_tmp[0:-pday]
print(data_tmp1)

print(data_pre)

# concat new columns
data_tmp = pd.concat([data_tmp1, data_tmp2], axis=1)
#save to disk
data_tmp.to_csv("stock_nor.csv", index=False)

df = pd.read_csv('stock_nor.csv')
print(df)

# 訓練神經網路
x_train = df.iloc[0:300, 0:5] 
y_train = df.iloc[0:300, 5]  
x_test = df.iloc[300:400, 0:5]  
y_test = df.iloc[300:400, 5]

# 建置神經網路
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(400, input_dim=5, activation="tanh"))
model.add(Dense(200, input_dim=5, activation="tanh"))
model.add(Dense(1, activation="tanh"))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
model.fit(x_train, y_train, epochs=1000, batch_size=200)
score = model.evaluate(x_test, y_test)


#儲存與載入神經網路MODEL
model.save("nn_stock.h5")

from keras.models import load_model
model = load_model("nn_stock.h5")

# 預測神經網路

y_pre = model.predict(x_test)

#還原標準化
data_min = np.min(data)
data_max = np.max(data)
data_mean = np.mean(data)

y_pre_org = y_pre*(data_max['close']-data_min['close']) + data_mean['close']
y_pre_org

#繪製實際與預測的股市行情
y_pre_org.shape
y_pre_org.reshape(99)

Y_org = data["close"]
y_test_org = Y_org[300:399]

y_test_org

# show results
line_1 = plt.plot(y_test_org.index, y_test_org, 'b', label='actual')
line_2 = plt.plot(y_test_org.index, y_pre_org, 'r--', label='predicted')
plt.ylabel('Value')
plt.xlabel('day')
plt.legend()
plt.show()