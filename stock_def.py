def stockpredict(num,days):
  data = pdr.get_data_yahoo(str(num) + ".TW")

  for i in range(len(data)):
    if (data.Volume[i] == '0'):
        data = data.drop(i)

  df = pd.DataFrame()
  df["volume"] = data["Volume"][-days:]
  df["open"] = data["Open"][-days:]
  df["high"] = data["High"][-days:]
  df["low"] = data["Low"][-days:]
  df["close"] = data["Close"][-days:]

  df_norm = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

  pday = 1
  data_tmp = df_norm.copy()

  data_tmp2 = data_tmp['close'].iloc[pday:]
  data_tmp2 = data_tmp2.reset_index(drop=True)
  data_tmp2.name = 'close after ' + str(pday) + ' days'

  data_pre = data_tmp[-pday:]  #
  data_tmp1 = data_tmp[0:-pday]

  data_tmp1 = data_tmp1.reset_index(drop=True)
  data_tmp = pd.concat([data_tmp1, data_tmp2], axis=1)

  X = data_tmp.iloc[:,0:5]
  Y = data_tmp.iloc[:,5]

  model = Sequential()
  model.add(Dense(200, input_dim=5, activation='tanh'))
  model.add(Dense(100, activation='tanh'))
  model.add(Dense(1, activation='tanh'))
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

  model.fit(X, Y, epochs=100, batch_size=200, verbose=0)

  p = model.predict(data_pre)

  data_min = np.min(df)
  data_max = np.max(df)
  data_mean = np.mean(df)

  predictions_org = p*(data_max['close']-data_min['close']) + data_mean['close']

  pred = predictions_org[-pday:]

  return pred[0][0]