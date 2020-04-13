import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import random
from sklearn.metrics import mean_squared_error
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

def generate_graphs(model, look_back, df, sep1, sep2):
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
    test_score = np.sqrt(np.mean((test_y[0] - test_predict1[:,0])**2))

    predictions = np.concatenate([train_predict, validate_predict, test_predict], axis=None)
    actual = np.concatenate([train_y, valid_y, test_y], axis=None)
    print(test_score)
    plt.plot(predictions)
    plt.plot(actual)
    plt.show()

    plt.plot(predictions-actual)
    plt.show()


dataset = 'SPX'
data_folder = '../Data/'
df = pd.read_csv(data_folder + dataset + '_Index_processed.csv', index_col = 0)
df['Realized Volatility'], mean, std = normalize(df['Realized Volatility'])

train_percent = 0.8
validate_percent = 0.1
epochs = 50

sep1 = round(train_percent * len(df))
sep2 = round((train_percent + validate_percent) * len(df))

best_model1 = None
best_model2 = None
best_model3 = None

for look_back in [1, 5, 22, 30]:
    train_data = df[0:sep1]
    validate_data = df[sep1-look_back:sep2]
    test_data = df[sep2-look_back:]

    for num_neurons in [32, 64, 128, 256, 512]:
        print(look_back, num_neurons)

        train_x, train_y = create_tensor(train_data, look_back)
        valid_x, valid_y = create_tensor(validate_data, look_back)

        random.seed(1)
        model1 = Sequential()
        model1.add(LSTM(num_neurons, input_shape=(1, look_back)))
        model1.add(Dropout(0.5))
        model1.add(Dense(1))
        model1.compile(loss='mean_squared_error', optimizer='adam')
        model1.fit(train_x, train_y, epochs=epochs, batch_size=4, verbose=2)

        random.seed(1)
        model2 = Sequential()
        model2.add(LSTM(num_neurons, input_shape=(1, look_back), return_sequences = True))
        model2.add(Dropout(0.5))
        model2.add(LSTM(num_neurons))
        model2.add(Dropout(0.5))
        model2.add(Dense(1))
        model2.compile(loss='mean_squared_error', optimizer='adam')
        model2.fit(train_x, train_y, epochs=epochs, batch_size=4, verbose=2)

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
        model3.fit(train_x, train_y, epochs=epochs, batch_size=4, verbose=2)

        train_y = denormalize(train_y, mean, std)
        valid_y = denormalize(valid_y, mean, std)

        train_predict1 = model1.predict(train_x)
        train_predict1 = denormalize(train_predict1, mean, std)
        validate_predict1 = model1.predict(valid_x)
        validate_predict1 = denormalize(validate_predict1, mean, std)


        train_score1 = np.sqrt(np.mean((train_y[0] - train_predict1[:,0])**2))
        validate_score1 = np.sqrt(np.mean((valid_y[0] - validate_predict1[:,0])**2))

        if best_model1 is None:
            best_model1 = (model1, validate_score1, look_back, num_neurons)
        elif best_model1[1] > validateScore1:
            best_model1 = (model1, validate_score1, look_back, num_neurons)
            
        train_predict2 = model2.predict(train_x)
        train_predict2 = denormalize(train_predict2, mean, std)
        validate_predict2 = model2.predict(valid_x)
        validate_predict2 = denormalize(validate_predict2, mean, std)

        train_score2 = np.sqrt(np.mean((train_y[0] - train_predict2[:,0])**2))
        validate_score2 = np.sqrt(np.mean((valid_y[0] - validate_predict2[:,0])**2))

        if best_model2 is None:
            best_model2 = (model2, validate_score2, look_back, num_neurons)
        elif best_model2[1] > validateScore2:
            best_model2 = (model2, validate_score2, look_back, num_neurons)

        train_predict3 = model3.predict(train_x)
        train_predict3 = denormalize(train_predict3, mean, std)
        validate_predict3 = model3.predict(valid_x)
        validate_predict3 = denormalize(validate_predict3, mean, std)

        train_score3 = np.sqrt(np.mean((train_y[0] - train_predict3[:,0])**2))
        validate_score3 = np.sqrt(np.mean((valid_y[0] - validate_predict3[:,0])**2))

        if best_model3 is None:
            best_model3 = (model3, validate_score3, look_back, num_neurons)
        elif best_model3[1] > validateScore1:
            best_model3 = (model3, validate_score3, look_back, num_neurons)


generate_graphs(best_model1[0], best_model1[2], df, sep1, sep2)
generate_graphs(best_model2[0], best_model2[2], df, sep1, sep2)
generate_graphs(best_model3[0], best_model3[2], df, sep1, sep2)

