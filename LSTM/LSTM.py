#%env PYTHONHASHSEED=0
# for running inside notebook, otherwise set hash seed manually
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
import os
import keras

def create_tensor(dataset, look_back = 1): #transforms dataframe into inputs for Keras LSTM
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset.iloc[i:(i+look_back), 0]
        data_x.append(a)
        data_y.append(dataset.iloc[i + look_back, 0])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_x = np.reshape(data_x, (data_x.shape[0], 1, data_x.shape[1]))
    return data_x, data_y

def normalize(series):
    return (series - series.mean())/series.std(), series.mean(), series.std()

def denormalize(series, mean, std):
    return series*std + mean

def create_csv(model, look_back, df, sep1, sep2, layers):
    train_data = df[0:sep1]
    validate_data = df[sep1-look_back:sep2]
    test_data = df[sep2-look_back:]

    test_x, test_y = create_tensor(test_data, look_back)
    test_y = denormalize(test_y, mean, std)

    test_predict = model.predict(test_x)
    test_predict = denormalize(test_predict, mean, std)
    result = test_y - test_predict[:, 0]
    c = df[-len(test_predict):].copy()
    c['Realized Volatility'] = result
    c.to_csv(f'Residuals_{layers}layer.csv')

def generate_graphs(model, look_back, df, sep1, sep2, history, layers, Type):
    #plot graphs for data
    train_data = df[0:sep1]
    validate_data = df[sep1-look_back:sep2]
    test_data = df[sep2-look_back:]

    train_x, train_y = create_tensor(train_data, look_back)
    valid_x, valid_y = create_tensor(validate_data, look_back)
    train_y = denormalize(train_y, mean, std)
    valid_y = denormalize(valid_y, mean, std)
    test_x, test_y = create_tensor(test_data, look_back)
    test_y = denormalize(test_y, mean, std)

    train_predict = model.predict(train_x)
    train_predict = denormalize(train_predict, mean, std)
    validate_predict = model.predict(valid_x)
    validate_predict = denormalize(validate_predict, mean, std)
    test_predict = model.predict(test_x)
    test_predict = denormalize(test_predict, mean, std)
    test_score = np.sqrt(np.mean((test_y - test_predict[:,0])**2))

    predictions = np.concatenate([train_predict, validate_predict, test_predict], axis=None)
    actual = np.concatenate([train_y, valid_y, test_y], axis=None)
    print(test_score)
    print(mean_absolute_error(test_y, test_predict[:,0]))
    print(explained_variance_score(test_y, test_predict[:,0]))
    plt.plot(predictions, linewidth=0.5, label = 'Predicted values')
    plt.plot(actual, linewidth=0.5, alpha=0.7, label = 'Actual values')
    plt.axvline(x=sep1, color = 'black')
    plt.axvline(x=sep2, color = 'black')
    plt.xlabel('Observation')
    plt.ylabel('Volatility')
    plt.title(f'{Type} LSTM Forecast: {layers} Layers, {look_back} Lags')
    plt.legend()
    plt.show()

    residuals = actual-predictions
    test_residuals = test_y - test_predict[:, 0]
    print(test_residuals.mean()/test_y.mean())
    print(test_residuals.std())
    plt.hist(test_residuals)
    plt.ylabel('Frequency')
    plt.xlabel('Volatility')
    plt.title(f'{Type} Volatility frequency: {layers} Layers, {look_back} Lags')
    plt.show()

    standardized = (test_residuals - test_residuals.mean()) / test_residuals.std()
    plt.plot(standardized, linewidth=0.5)
    plt.axhline(y=2, color = 'red')
    plt.axhline(y=-2, color = 'red')
    plt.title(f'{Type} Standardized LSTM Test Set Residuals: {layers} Layers, {look_back} Lags')
    plt.xlabel('Observation')
    plt.ylabel('Residuals')
    plt.show()
    print(standardized[abs(standardized)>1].count()/len(standardized))
    print(standardized[abs(standardized)>2].count()/len(standardized))

    plt.plot(residuals, linewidth=0.5)
    plt.axvline(x=sep1, color = 'black')
    plt.axvline(x=sep2, color = 'black')
    plt.title(f'{Type} LSTM Residuals: {layers} Layers, {look_back} Lags')
    plt.xlabel('Observation')
    plt.ylabel('Residuals')
    plt.show()

    plt.plot(history.history['loss'])
    plt.title(f'{Type} Training MSE: {layers} Layers, {look_back} Lags')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.show()

dataset = 'SPX'
vol_type = 'LSPD'
#vol_type = 'RV'
file_end = vol_type if vol_type = 'LSPD' else 'processed
df = pd.read_csv(dataset + '_Index_' + file_end + '.csv', index_col = 0)
df['Realized Volatility'], mean, std = normalize(df['Realized Volatility'])

# 1 thread for reproducibility
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                        inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

train_percent = 0.8
validate_percent = 0.1
epochs = 50
batch_size = 4

sep1 = round(train_percent * len(df))
sep2 = round((train_percent + validate_percent) * len(df))

best_model1 = None
best_model2 = None
best_model3 = None

results1 = {1:{}, 5:{}, 22:{}, 29:{}}
results2 = {1:{}, 5:{}, 22:{}, 29:{}}
results3 = {1:{}, 5:{}, 22:{}, 29:{}}

for look_back in [1, 5, 22, 29]:
    #split data
    train_data = df[0:sep1]
    validate_data = df[sep1-look_back:sep2]
    test_data = df[sep2-look_back:]

    for num_neurons in [32, 64, 128, 256]:
        print(look_back, num_neurons)

        train_x, train_y = create_tensor(train_data, look_back)
        valid_x, valid_y = create_tensor(validate_data, look_back)

        #reproducibility
        random.seed(1)
        os.environ['PYTHONHASHSEED']=str(1)
        np.random.seed(1)
        tf.random.set_seed(1)

        #create 1-layer model
        model1 = Sequential()
        model1.add(LSTM(num_neurons, input_shape=(1, look_back),
                   kernel_initializer=keras.initializers.glorot_uniform(seed=1),
                   bias_initializer=keras.initializers.Constant(value=0.)))
        model1.add(Dropout(0.5, seed=1))
        model1.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed=1),
                   bias_initializer=keras.initializers.Constant(value=0.)))
        model1.compile(loss='mean_squared_error', optimizer='adam')
        history1 = model1.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2)

        random.seed(1)
        os.environ['PYTHONHASHSEED']=str(1)
        np.random.seed(1)
        tf.random.set_seed(1)
        model2 = Sequential()
        model2.add(LSTM(num_neurons, input_shape=(1, look_back), return_sequences = True,
                   kernel_initializer=keras.initializers.glorot_uniform(seed=1),
                   bias_initializer=keras.initializers.Constant(value=0.)))
        model2.add(Dropout(0.5, seed=1))
        model2.add(LSTM(num_neurons, kernel_initializer=keras.initializers.glorot_uniform(seed=1),
                   bias_initializer=keras.initializers.Constant(value=0.)))
        model2.add(Dropout(0.5, seed=2))
        model2.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed=1),
                   bias_initializer=keras.initializers.Constant(value=0.)))
        model2.compile(loss='mean_squared_error', optimizer='adam')
        history2 = model2.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2)

        random.seed(1)
        os.environ['PYTHONHASHSEED']=str(1)
        np.random.seed(1)
        tf.random.set_seed(1)
        model3 = Sequential()
        model3.add(LSTM(num_neurons, input_shape=(1, look_back), return_sequences = True,
                   kernel_initializer=keras.initializers.glorot_uniform(seed=1),
                   bias_initializer=keras.initializers.Constant(value=0.)))
        model3.add(Dropout(0.5, seed=1))
        model3.add(LSTM(num_neurons, return_sequences = True,
                   kernel_initializer=keras.initializers.glorot_uniform(seed=1),
                   bias_initializer=keras.initializers.Constant(value=0.)))
        model3.add(Dropout(0.5, seed=2))
        model3.add(LSTM(num_neurons, kernel_initializer=keras.initializers.glorot_uniform(seed=1),
                   bias_initializer=keras.initializers.Constant(value=0.)))
        model3.add(Dropout(0.5, seed=3))
        model3.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed=1),
                   bias_initializer=keras.initializers.Constant(value=0.)))
        model3.compile(loss='mean_squared_error', optimizer='adam')
        history3 = model3.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2)

        #denormalize validation data for calculating accuracy
        valid_y = denormalize(valid_y, mean, std)

        validate_predict1 = model1.predict(valid_x)
        validate_predict1 = denormalize(validate_predict1, mean, std)

        #RMSE is accuracy
        validate_score1 = np.sqrt(mean_squared_error(valid_y, validate_predict1[:, 0]))
        results1[look_back][num_neurons] = validate_score1

        #adjust best model
        if best_model1 is None:
            best_model1 = (model1, validate_score1, look_back, num_neurons, history1)
        elif best_model1[1] > validate_score1:
            best_model1 = (model1, validate_score1, look_back, num_neurons, history1)
            
        validate_predict2 = model2.predict(valid_x)
        validate_predict2 = denormalize(validate_predict2, mean, std)

        validate_score2 = np.sqrt(mean_squared_error(valid_y, validate_predict2[:, 0]))
        results2[look_back][num_neurons] = validate_score2

        if best_model2 is None:
            best_model2 = (model2, validate_score2, look_back, num_neurons, history2)
        elif best_model2[1] > validate_score2:
            best_model2 = (model2, validate_score2, look_back, num_neurons, history2)

        validate_predict3 = model3.predict(valid_x)
        validate_predict3 = denormalize(validate_predict3, mean, std)

        validate_score3 = np.sqrt(mean_squared_error(valid_y, validate_predict3[:, 0]))

        if best_model3 is None:
            best_model3 = (model3, validate_score3, look_back, num_neurons, history3)
        elif best_model3[1] > validate_score3:
            best_model3 = (model3, validate_score3, look_back, num_neurons, history3)
        results3[look_back][num_neurons] = validate_score3

print(results1)
print(results2)
print(results3)
print(best_model1)
print(best_model2)
print(best_model3)

#generate graphs for training data on best models
plt.rcParams['figure.figsize'] = [12, 6]
generate_graphs(best_model1[0], best_model1[2], df, sep1, sep2, best_model1[4], 1, vol_type)
print('----------------------------')
generate_graphs(best_model2[0], best_model2[2], df, sep1, sep2, best_model2[4], 2, vol_type)
print('----------------------------')
generate_graphs(best_model3[0], best_model3[2], df, sep1, sep2, best_model3[4], 3, vol_type)
