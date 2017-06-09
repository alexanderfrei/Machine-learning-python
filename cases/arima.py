import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os

os.chdir("C:/Users/Aleksandr.Turutin/Downloads/")
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

data = pd.read_csv('AirPassengers.csv')

# create index
data = data.set_index(pd.DatetimeIndex(data['Month']))
data.index.name = None
data = data.drop('Month', axis=1)


# slicing
ts = data
ts.loc['1949-01-01'], ts.loc['1949-01-01':'1949-05-01']  # slicing by range
ts.loc['1949']  # slicing by year

plt.plot(ts)

# Check stationarity
from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(center=False, window=12).mean()
    rolstd = timeseries.rolling(center=False, window=12).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = {}
    for col in timeseries.columns.values:
        dftest[col] = adfuller(timeseries[col])
    dfoutput = pd.Series(dftest['#Passengers'][0:4],
                         index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest['#Passengers'][4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


test_stationarity(ts)  # p-value > 0.05 non-significant, timeseries not stationary

# trend
# Aggregation – taking average for a time period like monthly/weekly averages
# Smoothing – taking rolling averages
# Polynomial Fitting – fit a regression model

ts_log = np.log(ts)
moving_avg = ts_log.rolling(window=12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')


