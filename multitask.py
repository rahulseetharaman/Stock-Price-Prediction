#Importing the libraries

import datetime as dt
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
#Setting start and end dates and fetching the historical data
df = pd.read_csv('NSE.csv')

#Visualizing the fetched data
# plt.figure(figsize=(14,14))
# plt.plot(df['Close'])
# plt.title('Historical Stock Value')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.show()

#Data Preprocessing
df['Date'] = df.index
data2 = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
data2['Date'] = df['Date']
data2['Open'] = df['Open']
data2['High'] = df['High']
data2['Low'] = df['Low']
data2['Close'] = df['Close']
train_set_c = data2.iloc[:, 4:].values
train_set_o = data2.iloc[:,1:2].values
train_set_h = data2.iloc[:,2:3].values
train_set_l = data2.iloc[:,3:4].values

sc1 = MinMaxScaler(feature_range = (0, 1))
sc2 = MinMaxScaler(feature_range = (0, 1))
sc3 = MinMaxScaler(feature_range = (0, 1))
sc4 = MinMaxScaler(feature_range = (0, 1))
training_set_scaled_c = sc1.fit_transform(train_set_c)
training_set_scaled_o = sc2.fit_transform(train_set_o)
training_set_scaled_h = sc3.fit_transform(train_set_h)
training_set_scaled_l = sc4.fit_transform(train_set_l)
X_train_c = []
X_train_o = []
X_train_h = []
X_train_l = []
y_train_o = []
y_train_h = []
y_train_l = []
y_train_c = []
y_train = []
for i in range(60, len(train_set_c)):
    X_train_c.append(training_set_scaled_c[i-60:i, 0])
    X_train_o.append(training_set_scaled_o[i-60:i, 0])
    X_train_h.append(training_set_scaled_h[i-60:i, 0])
    X_train_l.append(training_set_scaled_l[i-60:i, 0])
    # l = []
    # l.extend(training_set_scaled_c[i, 0])
    # l.extend(training_set_scaled_o[i, 0])
    # l.extend(training_set_scaled_h[i, 0])
    # l.extend(training_set_scaled_l[i, 0])
    # y_train.append(l)
    y_train_o.append(training_set_scaled_o[i, 0])
    y_train_h.append(training_set_scaled_h[i, 0])
    y_train_l.append(training_set_scaled_l[i, 0])
    y_train_c.append(training_set_scaled_c[i, 0])

X_train_c,X_train_o,X_train_h,X_train_l = np.array(X_train_c),np.array(X_train_o),np.array(X_train_h),np.array(X_train_l)
X_train_c = np.reshape(X_train_c, (X_train_c.shape[0], X_train_c.shape[1], 1))
X_train_o = np.reshape(X_train_o, (X_train_o.shape[0], X_train_o.shape[1], 1))
X_train_h = np.reshape(X_train_h, (X_train_h.shape[0], X_train_h.shape[1], 1))
X_train_l = np.reshape(X_train_l, (X_train_l.shape[0], X_train_l.shape[1], 1))



from keras.models import Sequential,Model
from keras.layers import  Dense,Input,Concatenate
from keras.layers import LSTM,Embedding,Bidirectional

input_tensor_c = Input(shape=( X_train_c.shape[1],1), dtype='float32')
x_c = Bidirectional(LSTM(128, return_sequences=True))(input_tensor_c)
x_c = Dropout(0.2)(x_c)
x_c = Bidirectional(LSTM(64, return_sequences=False))(x_c)
x_c = Dropout(0.2)(x_c)
output_tensor_c = Dense(1)(x_c)
regressor_c = Model(input_tensor_c, output_tensor_c)

input_tensor_o = Input(shape=( X_train_o.shape[1],1), dtype='float32')
x_o = Bidirectional(LSTM(128, return_sequences=True))(input_tensor_o)
x_o = Dropout(0.2)(x_o)
x_o = Bidirectional(LSTM(64, return_sequences=False))(x_o)
x_o = Dropout(0.2)(x_o)
output_tensor_o = Dense(1)(x_o)
regressor_o = Model(input_tensor_o, output_tensor_o)


input_tensor_h = Input(shape=( X_train_h.shape[1],1), dtype='float32')
x_h = Bidirectional(LSTM(128, return_sequences=True))(input_tensor_h)
x_h = Dropout(0.2)(x_h)
x_h = Bidirectional(LSTM(64, return_sequences=False))(x_h)
x_h = Dropout(0.2)(x_h)
output_tensor_h = Dense(1)(x_h)
regressor_h = Model(input_tensor_h, output_tensor_h)


input_tensor_l = Input(shape=( X_train_l.shape[1],1), dtype='float32')
x_l = Bidirectional(LSTM(128, return_sequences=True))(input_tensor_l)
x_l = Dropout(0.2)(x_l)
x_l = Bidirectional(LSTM(64, return_sequences=False))(x_l)
x_l = Dropout(0.2)(x_l)
output_tensor_l = Dense(1)(x_l)
regressor_l = Model(input_tensor_l, output_tensor_l)

combined = Concatenate()([regressor_c.output, regressor_o.output,regressor_h.output,regressor_l.output])
mix2= Dense(100)(combined)
mix2 = Dense(50)(mix2)

out1 = Dense(1)(mix2)
out2 = Dense(1)(mix2)
out3 = Dense(1)(mix2)
out4 = Dense(1)(mix2)

regressor = Model(inputs=[input_tensor_c,input_tensor_o,input_tensor_h,input_tensor_l],outputs=[out1,out2,out3,out4])
# #Compiling and fitting the model
regressor.compile(optimizer = "rmsprop", loss = 'mean_squared_error')
regressor.fit([X_train_c,X_train_o,X_train_h,X_train_l], [np.array(y_train_c),np.array(y_train_o),np.array(y_train_h),np.array(y_train_l)], epochs = 5, batch_size = 32)

regressor.save("multiBiLSTM.h5")

import tensorflow as tf
# from keras.models import load_model

#Fetching the test data and preprocessing
testdataframe = pd.read_csv("NSE.csv")
testdataframe['Date'] = testdataframe.index
testdata = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
testdata['Date'] = testdataframe['Date']
testdata['Open'] = testdataframe['Open']
testdata['High'] = testdataframe['High']
testdata['Low'] = testdataframe['Low']
testdata['Close'] = testdataframe['Close']


dataset_total_c = pd.concat((data2['Close'], testdata['Close']), axis = 0)
dataset_total_o = pd.concat((data2['Close'], testdata['Close']), axis = 0)
dataset_total_h = pd.concat((data2['Close'], testdata['Close']), axis = 0)
dataset_total_l = pd.concat((data2['Close'], testdata['Close']), axis = 0)
inputs_c = dataset_total_c[len(dataset_total_c) - len(testdata) - 60:].values
inputs_c = inputs_c.reshape(-1,1)
inputs_c = sc1.transform(inputs_c)

inputs_o = dataset_total_o[len(dataset_total_o) - len(testdata) - 60:].values
inputs_o = inputs_o.reshape(-1,1)
inputs_o = sc2.transform(inputs_o)

inputs_h = dataset_total_h[len(dataset_total_h) - len(testdata) - 60:].values
inputs_h = inputs_h.reshape(-1,1)
inputs_h = sc3.transform(inputs_h)

inputs_l = dataset_total_l[len(dataset_total_l) - len(testdata) -60:].values
inputs_l = inputs_l.reshape(-1,1)
inputs_l = sc4.transform(inputs_l)
X_test_c = []
X_test_o = []
X_test_h = []
X_test_l = []

y_test_o = []
y_test_h = []
y_test_l = []
y_test_c = []
for i in range(60, testdata.shape[0]):
    X_test_c.append(inputs_c[i-60:i, 0])
    X_test_o.append(inputs_o[i-60:i, 0])
    X_test_h.append(inputs_h[i-60:i, 0])
    X_test_l.append(inputs_l[i-60:i, 0])
    y_test_o.append(inputs_o[i,0])
    y_test_h.append(inputs_h[i,0])
    y_test_l.append(inputs_l[i,0])
    y_test_c.append(inputs_c[i,0])
X_test_c = np.array(X_test_c)
X_test_o = np.array(X_test_o)
X_test_h = np.array(X_test_h)
X_test_l = np.array(X_test_l)
y_test_o = np.array(y_test_o)
y_test_h = np.array(y_test_h)
y_test_l = np.array(y_test_l)
y_test_c = np.array(y_test_c)
X_test_c = np.reshape(X_test_c, (X_test_c.shape[0], X_test_c.shape[1], 1))
X_test_o = np.reshape(X_test_o, (X_test_o.shape[0], X_test_o.shape[1], 1))
X_test_h = np.reshape(X_test_h, (X_test_h.shape[0], X_test_h.shape[1], 1))
X_test_l = np.reshape(X_test_l, (X_test_l.shape[0], X_test_l.shape[1], 1))

regressor = load_model("multiBiLSTM.h5")
# regressor.compile(optimizer = "rmsprop", loss = 'mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
regressor.compile(optimizer = "rmsprop", loss = 'mean_squared_error')
#
# print(regressor.evaluate([X_test_c,X_test_o,X_test_h,X_test_l],[Y_test_c,Y_test_o,Y_test_h,Y_test_l]))
# regressor.compile(optimizer = "rmsprop",loss = 'mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError()])
#
# print(regressor.evaluate([X_test_c,X_test_o,X_test_h,X_test_l],[Y_test_c,Y_test_o,Y_test_h,Y_test_l]))
# regressor.compile(optimizer = "rmsprop",loss = 'mean_squared_error',metrics=[tf.keras.metrics.MeanAbsoluteError()])
#
# print(regressor.evaluate([X_test_c,X_test_o,X_test_h,X_test_l],[Y_test_c,Y_test_o,Y_test_h,Y_test_l]))


#Making predictions on the test data
predicted_stock_price = regressor.predict([X_test_c,X_test_o,X_test_h,X_test_l])
predicted_stock_price1 = sc1.inverse_transform(predicted_stock_price[0])
predicted_stock_price2 = sc2.inverse_transform(predicted_stock_price[1])
predicted_stock_price3 = sc3.inverse_transform(predicted_stock_price[2])
predicted_stock_price4 = sc4.inverse_transform(predicted_stock_price[3])

# Y_test_c = Y_test_o.reshape(-1,1)
y_test_c = sc1.inverse_transform(y_test_c.reshape(1,-1))

# Y_test_o = Y_test_h.reshape(-1,1)
y_test_o = sc2.inverse_transform(y_test_o.reshape(1,-1))

# Y_test_h = Y_test_l.reshape(-1,1)
y_test_h = sc3.inverse_transform(y_test_h.reshape(1,-1))

# Y_test_l = Y_test_c.reshape(-1,1)
y_test_l = sc4.inverse_transform(y_test_l.reshape(1,-1))

from sklearn.metrics import mean_squared_error,mean_absolute_error
y_test_tot = []
pred = []
y_test_tot.append(y_test_c)
y_test_tot.append(y_test_o)
y_test_tot.append(y_test_h)
y_test_tot.append(y_test_l)

pred.append(predicted_stock_price1)
pred.append(predicted_stock_price2)
pred.append(predicted_stock_price3)
pred.append(predicted_stock_price4)
y_test_c = np.array(y_test_c).reshape(-1,1)
y_test_o = np.array(y_test_o).reshape(-1,1)
y_test_h = np.array(y_test_h).reshape(-1,1)
y_test_l = np.array(y_test_l).reshape(-1,1)
y_test_tot = np.array(y_test_tot).reshape(-1,1)
pred = np.array(pred).reshape(-1,1)
print(y_test_c)
# print(predicted_stock_price)
rmse1=np.sqrt(np.mean(((predicted_stock_price1[50:]- y_test_c[50:])**2)))
rmse2=np.sqrt(np.mean(((predicted_stock_price2[50:]- y_test_o[50:])**2)))
rmse3=np.sqrt(np.mean(((predicted_stock_price3[50:]- y_test_h[50:])**2)))
rmse4=np.sqrt(np.mean(((predicted_stock_price4[50:]- y_test_l[50:])**2)))
print(mean_squared_error(predicted_stock_price1[50:],y_test_c[50:]))
print(mean_squared_error(predicted_stock_price2[50:],y_test_o[50:]))
print(mean_squared_error(predicted_stock_price3[50:],y_test_h[50:]))
print(mean_squared_error(predicted_stock_price4[50:],y_test_l[50:]))
print(rmse1)
print(rmse2)
print(rmse3)
print(rmse4)

print("***")
print(np.sqrt(mean_squared_error(y_test_tot[50:],pred[50:])))
print(mean_squared_error(y_test_tot[50:],pred[50:]))
print(mean_absolute_error(y_test_tot[50:],pred[50:]))
print(np.mean(np.abs((y_test_tot[50:] - pred[50:]) / y_test_tot[50:])) * 100)


plt.figure(figsize=(20,10))
plt.plot(y_test_o[50:], color = 'green', label = 'SBI Stock Open Price')
plt.plot(predicted_stock_price2[50:], color = 'red', label = 'Predicted SBI Stock Open Price')
plt.title('SBI Stock Open Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('SBI Stock Open Price')
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(y_test_h[50:], color = 'green', label = 'SBI Stock High Price')
plt.plot(predicted_stock_price3[50:], color = 'red', label = 'Predicted SBI Stock High Price')
plt.title('SBI Stock High Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('SBI Stock High Price')
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(y_test_l[50:], color = 'green', label = 'SBI Stock Low Price')
plt.plot(predicted_stock_price4[50:], color = 'red', label = 'Predicted SBI Stock Low Price')
plt.title('SBI Stock Low Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('SBI Stock Low Price')
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(y_test_c[50:], color = 'green', label = 'SBI Stock Close Price')
plt.plot(predicted_stock_price1[50:], color = 'red', label = 'Predicted SBI Stock Close Price')
plt.title('SBI Stock Close Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('SBI Stock Close Price')
plt.legend()
plt.show()