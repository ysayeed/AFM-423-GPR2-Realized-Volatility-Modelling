import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt

def create_tensor(dataset, look_back = 1):
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

def generate_graphs(model, look_back, df, sep1, sep2, history, layers, Type):
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

    plt.scatter(test_y, test_residuals)
    plt.title(f'{Type} Residuals vs Volatility: {layers} Layers, {look_back} Lags')
    plt.ylabel('Residuals')
    plt.xlabel('Volatility')
    plt.show()

    past_1sd = np.count_nonzero(test_residuals > test_residuals.std()) + np.count_nonzero(test_residuals < -test_residuals.std())
    print(past_1sd/len(test_residuals))
    past_2sd = np.count_nonzero(test_residuals > 2*test_residuals.std()) + np.count_nonzero(test_residuals < -2*test_residuals.std())
    print(past_2sd/len(test_residuals))
    plt.plot(test_residuals, linewidth=0.5)
    plt.axhline(y=2*test_residuals.std(), color = 'red')
    plt.axhline(y=-2*test_residuals.std(), color = 'red')
    plt.title(f'{Type} Test LSTM Residuals: {layers} Layers, {look_back} Lags')
    plt.xlabel('Observation')
    plt.ylabel('Residuals')
    plt.show()

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
vol_type = 'processed' # 'LSPD'
df = pd.read_csv(dataset + '_Index_'+vol_type+'.csv', index_col = 0)
df['Realized Volatility'], mean, std = normalize(df['Realized Volatility'])

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
    train_data = df[0:sep1]
    validate_data = df[sep1-look_back:sep2]
    test_data = df[sep2-look_back:]

    for num_neurons in [32, 64, 128, 256]:
        print(look_back, num_neurons)

        train_x, train_y = create_tensor(train_data, look_back)
        valid_x, valid_y = create_tensor(validate_data, look_back)

        random.seed(1)
        model1 = Sequential()
        model1.add(LSTM(num_neurons, input_shape=(1, look_back)))
        model1.add(Dropout(0.5))
        model1.add(Dense(1))
        model1.compile(loss='mean_squared_error', optimizer='adam')
        history1 = model1.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2)

        random.seed(1)
        model2 = Sequential()
        model2.add(LSTM(num_neurons, input_shape=(1, look_back), return_sequences = True))
        model2.add(Dropout(0.5))
        model2.add(LSTM(num_neurons))
        model2.add(Dropout(0.5))
        model2.add(Dense(1))
        model2.compile(loss='mean_squared_error', optimizer='adam')
        history2 = model2.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2)

        random.seed(1)
        model3 = Sequential()
        model3.add(LSTM(num_neurons, input_shape=(1, look_back), return_sequences = True))
        model3.add(Dropout(0.5))
        model3.add(LSTM(num_neurons, return_sequences = True))
        model3.add(Dropout(0.5))
        model3.add(LSTM(num_neurons))
        model3.add(Dropout(0.5))
        model3.add(Dense(1))
        model3.compile(loss='mean_squared_error', optimizer='adam')
        history3 = model3.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2)

        valid_y = denormalize(valid_y, mean, std)

        validate_predict1 = model1.predict(valid_x)
        validate_predict1 = denormalize(validate_predict1, mean, std)


        validate_score1 = np.sqrt(mean_squared_error(valid_y, validate_predict1[:, 0]))
        results1[look_back][num_neurons] = validate_score1

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

plt.rcParams['figure.figsize'] = [12, 6]
generate_graphs(best_model1[0], best_model1[2], df, sep1, sep2, best_model1[4], 1, 'RV')
print('----------------------------')
generate_graphs(best_model2[0], best_model2[2], df, sep1, sep2, best_model2[4], 2, 'RV')
print('----------------------------')
generate_graphs(best_model3[0], best_model3[2], df, sep1, sep2, best_model3[4], 3, 'RV')
