import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
import datetime

def timer(start_time=None):
    if not start_time:
        start_time=datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(),3600)
        tmin, tsec = divmod(temp_sec,60)
        print('\n Time Taken: %i Hours and %i Minutes and %i Seconds' % (thour, tmin, round(tsec,2)))

# X_final is the data along with the variable that we need to forecast
# https://www.youtube.com/watch?v=tepxdcepTbY&t=538s
scaler = StandardScaler()
scaler = scaler.fit(X_final)
X_final1 = scaler.transform(X_final)

trainX = []
trainY = []
n_future = 1
n_past = 14

for i in range(n_past,len(X_final1)-n_future+1):
    trainX.append(X_final1[i -n_past:i, 0:X_final1.shape[1]])
    trainY.append(X_final1[i + n_future -1 : i + n_future, 0])
    
trainX, trainY = np.array(trainX), np.array(trainY)
                          
print('trainX shape =={}.', format(trainX.shape))
print('trainY shape =={}.', format(trainY.shape))

model = Sequential()
model.add(LSTM(64, activation = 'relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation = 'relu', return_sequences= False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer = 'adam', loss='mse')
model.summary()

history = model.fit(trainX, trainY, epochs = 10, batch_size = 16, validation_split = 0.2, verbose = 1)

plt.plot(history.history['loss'], label = 'Training loss')
plt.plot(history.history['val_loss'], label = 'Validation loss')
plt.legend()

n_future = 10
train_dates = X['Booking Date']
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods = n_future, freq = '1d').tolist()

forecast = model.predict(trainX[-n_future:])
forecast_copies = np.repeat(forecast, X_final1.shape[1], axis=1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]

forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Transaction Amount(Rs.)':y_pred_future})
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

original = X_final1[['Booking_Date', 'Transaction Amount(Rs.)']]
original['Booking_Date'] = pd.to_datetime(original['Booking_Date'])
original = original.loc[original['Booking_Date']>=somedate]

sns.lineplot(original['Booking_date'], original['Transaction Amount (Rs)'])
sns.lineplot(df_forecast['Booking_date'], df_forecast['Transaction Amount (Rs)'])
