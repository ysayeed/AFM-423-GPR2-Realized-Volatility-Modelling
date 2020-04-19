import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statistics
import statsmodels.api as sm
from math import sqrt


dataset = 'SPX'
data_folder = '../Data/'
pred = pd.read_csv(data_folder + dataset + '_Index_processed.csv', index_col = 0)
print(pred)

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
train = df[:9 * length//10]
test = df[9 * length//10:]

x_labels = [
            'Realized Volatility Prev Day', \
            'Realized Volatility Weekly', \
            'Realized Volatility Monthly' \
           ]

y_label = 'Realized Volatility'

# print(train)
# print(test)
# HAR model fit:
har = LinearRegression().fit(train[x_labels], train[y_label])
# print(har.coef_)
# print(har.intercept_)

# eval:
# in and out sample trend:
train_test_pred = har.predict(df[x_labels])
rmse = sqrt(mean_squared_error(df[y_label], train_test_pred))
print(rmse)

plt.xticks([], [])
plt.plot(train_test_pred, label = 'Predicted values')
plt.plot(df[y_label], label = 'True values')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title(f'HAR Forecast over True Value (Train and Test)')
plt.legend()
plt.savefig('har_train_test.png')
plt.clf()

# out sample:
test_pred = har.predict(test[x_labels])
rmse = sqrt(mean_squared_error(test[y_label], test_pred))
print(rmse)
plt.xticks([], [])
plt.plot(test_pred, label = 'Predicted values')
plt.plot(test[y_label], label = 'True values')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title(f'HAR Forecast over True Value (Test)')
plt.legend()
plt.savefig('har_test.png')
plt.clf()

# standardized residual
res = test[y_label] -test_pred

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
plt.title(f'HAR Standardized Residuals')
plt.savefig('har_res.png')
plt.clf()

# p-value of each?
mod = sm.OLS(train[y_label], train[x_labels])
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']
print(p_values)