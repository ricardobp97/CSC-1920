import pandas as pd
import numpy as np
import pydot
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

##########################################################################################################
########################################### INITIALIZING SESSION #########################################
##########################################################################################################

tf.random.set_seed(91195003)
np.random.seed(91190530)
tf.keras.backend.clear_session()

##########################################################################################################
############################################### PREPARE DATA #############################################
##########################################################################################################

def read_dataset(path=r'time_series_19-covid-Confirmed.csv'):
    df = pd.read_csv(path, nrows=1)
    columns = df.columns.tolist()
    cols_to_use = columns[:len(columns)-6]
    df = pd.read_csv(path, usecols=cols_to_use)
    return df

def prepare_dataset(df):
    df_aux = df.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'], inplace=False)
    df_aux = df_aux.sum().to_frame()
    df_aux.columns = ['cases']
    return df_aux

def data_normalization(df, norm_range=(-1,1)):
    scaler = MinMaxScaler(feature_range=norm_range)
    df[['cases']] = scaler.fit_transform(df[['cases']])
    return scaler

def to_supervised(df, timesteps):
    data = df.values
    X, y = list(), list()
    dataset_size = len(data)
    for curr_pos in range(dataset_size):
        input_index = curr_pos + timesteps
        label_index = input_index + 1
        if label_index < dataset_size:
            X.append(data[curr_pos:input_index, :])
            y.append(data[input_index:label_index, 0])
    return np.array(X).astype('float32'), np.array(y).astype('float32')

##########################################################################################################
############################################## LSTM MODEL ################################################
##########################################################################################################

def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred-y_true)))

def build_model(timesteps, features, h_neurons=64, activation='tanh', dropout_rate=0.5):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(h_neurons, return_sequences=True, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.LSTM(int(h_neurons*2), return_sequences=True, dropout=dropout_rate))
    model.add(tf.keras.layers.LSTM(int(h_neurons*4), return_sequences=False, dropout=dropout_rate))
    model.add(tf.keras.layers.Dense(h_neurons, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(
        loss=rmse,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae',rmse]
    )
    print(model.summary())
    tf.keras.utils.plot_model(model, 'covid19_model.png', show_shapes=True)
    return model

##########################################################################################################
############################################## PREDICTING ################################################
##########################################################################################################

def forecast(model, df, timesteps, multisteps, scaler):
    input_seq = df[-timesteps:].values
    inp = input_seq
    predictions = list()
    for step in range(1, multisteps+1):
        inp = inp.reshape(1, timesteps, 1)
        yhat = model.predict(inp, verbose=verbose)
        yhat_inversed = scaler.inverse_transform(yhat)
        predictions.append(yhat_inversed[0][0])
        inp = np.append(inp[0],yhat)
        inp = inp[-timesteps:]
    return predictions

def plot_forecast(data, predictions):
    plt.figure(figsize=(8,6))
    plt.plot(range(len(data)), data, color='green', label='Confirmed')
    plt.plot(range(len(data)-1, len(data) + len(predictions)-1), predictions, color='red', label='Prediction')
    plt.title('Confirmed Cases of COVID-19')
    plt.ylabel('Cases')
    plt.xlabel('Days')
    plt.legend()
    plt.show()

##########################################################################################################
############################################## EXECUTION #################################################
##########################################################################################################

timesteps = 5
univariate = 1
multisteps = 7
cv_splits = 3
epochs = 50
batch_size = 6
verbose = 1
df_countries = read_dataset()
df_world = prepare_dataset(df_countries)
df = df_world.copy()
scaler = data_normalization(df)
X, y = to_supervised(df,timesteps)
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=50, min_lr=0.00005)
model = build_model(timesteps, univariate)
model.fit(X,y,epochs=epochs,batch_size=batch_size,shuffle=False,verbose=verbose,callbacks=[lr])
predictions = forecast(model, df, timesteps, multisteps, scaler)
plot_forecast(df_world, predictions)