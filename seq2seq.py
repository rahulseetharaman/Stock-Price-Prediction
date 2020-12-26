import datetime as dt
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import *
from keras.models import *
from keras.optimizers import *
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
data2 = pd.DataFrame(columns = [ 'Open', 'High', 'Low', 'Close'])

data2['Open'] = df['Open']
data2['High'] = df['High']
data2['Low'] = df['Low']
data2['Close'] = df['Close']
train_set = data2.iloc[:,:].values

scalers=[]
from sklearn.preprocessing import *
for i in range(train_set.shape[1]):
    s=MinMaxScaler(feature_range = (0, 1))
    train_set[:,i:i+1]=s.fit_transform(train_set[:,i:i+1])
    scalers.append(s)
train_X=[]
train_Y=[]
for i in range(60, len(train_set)):

        train_X.append(train_set[i-60:i,:])
        train_Y.append(train_set[i-1:i,:])


X_train = np.array(train_X)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))


latent_dim=256

encoder_inputs =  Input(shape=( X_train.shape[1],4), dtype='float32')
encoder = Bidirectional(LSTM(latent_dim, return_state=True))

encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]


# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_inputs = Input(shape=(1,4))
decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(4,activation='relu')
decoder_outputs = decoder_dense(decoder_outputs)
regressor = Model([encoder_inputs, decoder_inputs], decoder_outputs)

regressor.compile(optimizer = "rmsprop", loss = 'mean_squared_error')
encoder_input_data=X_train
print(X_train.shape)
decoder_target_data=np.array(train_Y).astype(np.float32)
print(decoder_target_data.shape)
decoder_input_data = np.zeros(decoder_target_data.shape)
print(decoder_input_data.shape)
regressor.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          epochs=10)

model.save("seq2seq.h5")

testdataframe = pd.read_csv("NSE.csv")
testdataframe['Date'] = testdataframe.index
testdata = pd.DataFrame(columns = [ 'Open', 'High', 'Low', 'Close'])

testdata['Open'] = testdataframe['Open']
testdata['High'] = testdataframe['High']
testdata['Low'] = testdataframe['Low']
testdata['Close'] = testdataframe['Close']

dataset_total = pd.concat((data2,testdata),axis=0)


print(dataset_total)
inputs = dataset_total[len(dataset_total) - len(testdata) - 60:].values
print(inputs)
print(inputs.shape)
# inputs= inputs.reshape(-1,1)
# print(inputs.shape)

for i in range(inputs.shape[1]):
    inputs[:,i:i+1]=scalers[i].transform(inputs[:,i:i+1])

test_X=[]
test_Y=[]
for i in range(60, len(inputs)):

        test_X.append(inputs[i-60:i,:])
        test_Y.append(inputs[i-1:i,:])

X_test = np.array(test_X)
print(X_test.shape)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))

Y_test = np.array(test_Y).astype(np.float32)
decoder_output_target = Y_test
decoder_output_data = np.zeros(decoder_output_target.shape)

regressor = load_model("seq2seq.h5")

regressor.compile(optimizer = "rmsprop", loss = 'mean_squared_error')

#
# print(regressor.evaluate([X_test,decoder_output_data],decoder_output_target))
# regressor.compile(optimizer = "rmsprop",loss = 'mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError()])
# print(regressor.evaluate([X_test,decoder_output_data],decoder_output_target))
# regressor.compile(optimizer = "rmsprop",loss = 'mean_squared_error',metrics=[tf.keras.metrics.MeanAbsoluteError()])
#
# print(regressor.evaluate([X_test,decoder_output_data],decoder_output_target))

predicted_stock_price = regressor.predict([X_test,decoder_output_data])
print(predicted_stock_price.shape)

predicted_stock_price = predicted_stock_price.reshape(predicted_stock_price.shape[0],4)
Y_test = Y_test.reshape(Y_test.shape[0],4)


print(Y_test)
for j in range(4):
    Y_test[:,j]=scalers[j].inverse_transform(Y_test[:,j].reshape(1,-1))

for j in range(4):
    predicted_stock_price[:,j]=scalers[j].inverse_transform(predicted_stock_price[:,j].reshape(1,-1))

# print(Y_test)
# print(predicted_stock_price)
from sklearn.metrics import mean_squared_error,mean_absolute_error
rmse1=np.sqrt(np.mean(((predicted_stock_price[50:,0]- Y_test[50:,0])**2)))
rmse2=np.sqrt(np.mean(((predicted_stock_price[50:,1]- Y_test[50:,1])**2)))
rmse3=np.sqrt(np.mean(((predicted_stock_price[50:,2]- Y_test[50:,2])**2)))
rmse4=np.sqrt(np.mean(((predicted_stock_price[50:,3]- Y_test[50:,3])**2)))
print(mean_squared_error(predicted_stock_price[50:,0],Y_test[50:,0]))
print(mean_squared_error(predicted_stock_price[50:,1],Y_test[50:,1]))
print(mean_squared_error(predicted_stock_price[50:,2],Y_test[50:,2]))
print(mean_squared_error(predicted_stock_price[50:,3], Y_test[50:,3]))
print(rmse1)
print(rmse2)
print(rmse3)
print(rmse4)
print("****")

print(np.sqrt(mean_squared_error(Y_test[50:],predicted_stock_price[50:])))
# print(mean_squared_error(Y_test[50:],predicted_stock_price[50:]))
# print(mean_absolute_error(Y_test[50:],predicted_stock_price[50:]))
# print(np.mean(np.abs((Y_test[50:] - predicted_stock_price[50:]) / Y_test[50:])) * 100)

plt.figure(figsize=(20,10))
plt.plot(Y_test[50:,0], color = 'green', label = 'SBI Stock Open Price')
plt.plot(predicted_stock_price[50:,0], color = 'red', label = 'Predicted SBI Stock Open Price')
plt.title('SBI Stock Open Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('SBI Stock Open Price')
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(Y_test[50:,1], color = 'green', label = 'SBI Stock High Price')
plt.plot(predicted_stock_price[50:,1], color = 'red', label = 'Predicted SBI Stock High Price')
plt.title('SBI Stock High Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('SBI Stock High Price')
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(Y_test[50:,2], color = 'green', label = 'SBI Stock Low Price')
plt.plot(predicted_stock_price[50:,2], color = 'red', label = 'Predicted SBI Stock Low Price')
plt.title('SBI Stock Low Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('SBI Stock Low Price')
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(Y_test[50:,3], color = 'green', label = 'SBI Stock Close Price')
plt.plot(predicted_stock_price[50:,3], color = 'red', label = 'Predicted SBI Stock Close Price')
plt.title('SBI Stock Close Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('SBI Stock Close Price')
plt.legend()
plt.show()