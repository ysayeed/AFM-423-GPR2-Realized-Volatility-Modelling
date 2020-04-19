import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt
import statistics
from math import sqrt

plt.rcParams['figure.figsize'] = [12, 6]

dataset = 'SPX'
data_folder = '../Data/'
pred = pd.read_csv(data_folder + dataset + '_Index_processed.csv', index_col = 0)
# print(pred)

# create new df for weekly and monthly vol
daily = pred.shift(1)
weekly = daily.rolling(5).mean()
monthly = daily.rolling(22).mean()

daily.rename(columns={'Realized Volatility': 'Realized Volatility Prev Day'}, inplace=True)
weekly.rename(columns={'Realized Volatility': 'Realized Volatility Weekly'}, inplace=True)
monthly.rename(columns={'Realized Volatility': 'Realized Volatility Monthly'}, inplace=True)

df = pred.merge(daily, left_index=True, right_index=True)
df = df.merge(weekly, left_index=True, right_index=True)
df = df.merge(monthly,left_index=True, right_index=True)
df.dropna(inplace=True)

# train test split
length = len(df)
train = df[:8 * length//10]
validation = df[8 * length//10: 9 * length//10]
test = df[9 * length//10:]

x_labels = [
            'Realized Volatility Prev Day', \
            'Realized Volatility Weekly', \
            'Realized Volatility Monthly' \
           ]

y_label = 'Realized Volatility'

# HAR model fit: pick best number of estimators:
har_bag = None
best_n = 0
lowest_test_rmse = 100
cv_mses = []
for i in range(1, 21):
    model = BaggingRegressor(base_estimator=LinearRegression(),
                               n_estimators=i, max_samples=1/3,random_state=0) \
             .fit(train[x_labels], train[y_label])

    rmse = sqrt(mean_squared_error(validation[y_label], model.predict(validation[x_labels])))
    cv_mses.append(rmse)
    # print(mse)
    if rmse < lowest_test_rmse:
        lowest_test_rmse = rmse
        har_bag = model
        best_n = i

print(best_n)

# eval:
# RMSE against i
print(cv_mses)

plt.xticks(list(range(0,21)), list(range(1,21)))
plt.xlabel('Number of iterations')
plt.ylabel('Validation RMSE')
plt.title(f'Bagged HAR Tuning')
plt.plot(cv_mses, color='orange')
plt.savefig('bag_tuning.png')
plt.clf()

# in and out sample trend:
train_test_pred = har_bag.predict(df[x_labels])
rmse = sqrt(mean_squared_error(df[y_label], train_test_pred))
print(rmse)
plt.xticks([], [])
plt.plot(train_test_pred, label = 'Predicted values')
plt.plot(df[y_label], label = 'True values')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title(f'Bagged HAR Forecast over True Value (Train, Validation and Test)')
plt.legend()
plt.savefig('bag_har_train_test.png')
plt.clf()

# out sample:
test_pred = har_bag.predict(test[x_labels])
rmse = sqrt(mean_squared_error(test[y_label], test_pred))
print(rmse)
plt.xticks([], [])
plt.plot(test_pred, label = 'Predicted values')
plt.plot(test[y_label], label = 'True values')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title(f'Bagged HAR Forecast over True Value (Test)')
plt.legend()
plt.savefig('bag_har_test.png')
plt.clf()

# standardized residual
res =  test[y_label]-test_pred

# print(res)
res_std = statistics.stdev(res)
# print(res_std)
res_mean = statistics.mean(res)
# print(res_mean)
res_adj = (res - res_mean)/res_std
plt.axhline(y=1.96)
plt.axhline(y=-1.96)
plt.xticks([], [])
plt.plot(res_adj, 'o')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title(f'Bagged HAR Standardized Residuals')
plt.savefig('bag_har_res.png')
plt.clf()