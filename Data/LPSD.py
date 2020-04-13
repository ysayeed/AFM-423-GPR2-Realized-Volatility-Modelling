import pandas as pd
import numpy as np
import sys

target = sys.argv[1] + '_unprocessed.csv'

ds = pd.read_csv(target)

adj_ds = ds[['Date', 'Close']].set_index('Date').iloc[::-1] #set index and reverse order
adj_ds.index = pd.to_datetime(adj_ds.index)
adj_ds['Log Return'] = np.log(adj_ds['Close']/adj_ds['Close'].shift(1)) # log returns
adj_ds.at[adj_ds.index[0], 'Log Return'] = np.log(1+ (ds.loc[0, 'Close'] - ds.loc[0, 'Open']) / ds.loc[0, 'Open']) #adjust for very first price from opening
log_return = adj_ds['Log Return']
log_return[log_return > 0] = 0 #LPSD only
adj_ds['Log Return^2'] = adj_ds['Log Return']**2
assert(not adj_ds.isna().sum().any()) #check for no nans

rv_data = adj_ds[['Log Return^2']].groupby(adj_ds.index.date).sum() #group by date
rv_data['Realized Volatility'] = np.sqrt(rv_data['Log Return^2']) #sqrt
rv_data.drop(['Log Return^2'], 1, inplace=True) #remove the column
rv_data.drop(rv_data.head(1).index, inplace=True) #remove the last day (only partial data)
rv_data.to_csv(target.replace('unprocessed', 'LPSD')) #write csv
