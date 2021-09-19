import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import time

dataframe = pd.read_csv('AMZN.csv')
FullData = dataframe[['Close']].values
sc = MinMaxScaler()
DataScaler = sc.fit(FullData)
X = DataScaler.transform(FullData)

X_samples = list()
y_samples = list()

NumberOfRows = len(X)
TimeSteps = 20
FutureTimeSteps = 5

for i in range(TimeSteps , NumberOfRows-FutureTimeSteps , 1):
    x_sample = X[i-TimeSteps:i]
    y_sample = X[i:i+FutureTimeSteps]
    X_samples.append(x_sample)
    y_samples.append(y_sample)
    
X_data = np.array(X_samples)
X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)
y_data = np.array(y_samples)
y_data = y_data.reshape(y_data.shape[0], y_data.shape[1])

TestingRecords = 5
X_train = X_data[:-TestingRecords]
X_test = X_data[-TestingRecords:]
y_train = y_data[:-TestingRecords]
y_test = y_data[-TestingRecords:]

Timesteps = X_train.shape[1]
TotalFeatures = X_train.shape[2]

regressor = Sequential()

regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (Timesteps, TotalFeatures), return_sequences = True))
regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (Timesteps, TotalFeatures), return_sequences = True))
regressor.add(LSTM(units = 5, activation = 'relu', return_sequences = False))

regressor.add(Dense(units = FutureTimeSteps))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

StartTime = time.time()
regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)
EndTime = time.time()
print("## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ##')

predicted_Price = regressor.predict(X_test)
predicted_Price = DataScaler.inverse_transform(predicted_Price)

orig = y_test
orig = DataScaler.inverse_transform(y_test)

print('Accuracy:', 100 - (100 * (abs(orig-predicted_Price)/orig)).mean())

for i in range(len(orig)):
    Prediction = predicted_Price[i]
    Original = orig[i]
    plt.plot(Prediction, color = 'blue', label = 'Predicted Volume')
    plt.plot(Original, color = 'lightblue', label = 'Original Value')
    plt.title('### Accuracy of the predictions:' + str(100 - (100 * (abs(Original-Prediction)/Original)).mean().round(2)) + '% ###')
    startDateIndex = (FutureTimeSteps * TestingRecords) - FutureTimeSteps * (i+1)
    endDateIndex = (FutureTimeSteps * TestingRecords) - FutureTimeSteps * (i+1) + FutureTimeSteps
    TotalRows = dataframe.shape[0]
    plt.xticks(range(FutureTimeSteps), dataframe.iloc[TotalRows - endDateIndex : TotalRows - (startDateIndex) , :]['Date'])
    plt.ylabel('Stock Price')
    
    plt.legend()
    fig = plt.gcf()
    fig.set_figwidth(20)
    fig.set_figheight(3)
    plt.show()
    
Last20Days = np.array([3631.199951,
3573.629883,
3549.590088,
3573.189941,
3585.199951,
3638.030029,
3656.639893,
3699.820068,
3626.389893,
3630.320068,
3599.919922,
3327.590088,
3331.479980,
3366.239990,
3354.719971,
3375.989990,
3344.939941,
3341.870117,
3320.679932,
3292.110107])

Last20Days = Last20Days.reshape(-1,1)
X_test = DataScaler.transform(Last20Days)
NumberofSamples = 1
TimeSteps = X_test.shape[0]
NumberofFeatures = X_test.shape[1]
X_test = X_test.reshape(NumberofSamples, TimeSteps, NumberofFeatures)

Next5DaysPrice = regressor.predict(X_test)
Next5DaysPrice = DataScaler.inverse_transform(Next5DaysPrice)
print(Next5DaysPrice)
