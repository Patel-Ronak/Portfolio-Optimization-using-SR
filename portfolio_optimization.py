# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 19:02:37 2021

@author: Ronak
"""


import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

import scipy.optimize as sci_opt

# List of stocks
stocks = ['AAPL', 'TSLA', 'AMZN', 'FB']
number_of_stocks = len(stocks)

# Portfolio Stats
start_val = 1000000
allocs = np.array([0.25, 0.25, 0.25, 0.25])

# Gather stock data
start = datetime.datetime(2020,1,1)
end = datetime.date.today()

df = pd.DataFrame()
for stock in stocks:
    df[stock] = pdr.get_data_yahoo(stock, start, end)['Adj Close']
  
# print(df.head())
    

# Plotting Data
title = 'Portfolio Adj. Close Price History'
port_stocks = df
for x in port_stocks.columns.values:
    plt.plot(port_stocks[x], label = x)
    
plt.title(title)
plt.xlabel('Date')
plt.ylabel('Adj. Close ($USD)')
plt.legend(port_stocks.columns.values, loc = 'upper left')
plt.show()

# Daily returns
returns = df.pct_change()

# Create and show annualized covariance matrix
cov_matrix_annual = returns.cov() * 252

# Calcualte portfolio variance, volatility, and annual return

port_var = np.dot(allocs.T, np.dot(cov_matrix_annual, allocs))

port_vol = np.sqrt(port_var)

port_ret = np.sum(returns.mean() * allocs * 252)

# Display values
per_var = str(round(port_var,2)*100) + '%'
per_vol = str(round(port_vol,2)*100) + '%'
per_ret = str(round(port_ret,2)*100) + '%'

print('Expected varaince: {}, Expected volality: {}, Expected return: {}'.format(per_var, per_vol, per_ret))

# Optimization Functions

def metrics(allocs):
    
    # Convert allocs to array
    allocs = np.array(allocs)
    
    # Calculate returns
    ret = np.sum(returns.mean() * allocs * 252)
    
    # Calculate variance
    var = np.dot(allocs.T, np.dot(cov_matrix_annual, allocs))
    
    # Calculate volatility
    vol = np.sqrt(var)
    
    # Calculate SR
    sr = ret/vol
    
    return np.array([ret,vol,sr])

def neg_sr(allocs):
    return metrics(allocs)[2] - 1

def volatility(allocs):
    return metrics(allocs)[1]
    
def check_sum(allocs):
    return np.sum(allocs)-1

# define boundaries and constraints
bounds = tuple((0,1) for stock in range(number_of_stocks))

constraints = ({'type': 'eq', 'fun': check_sum})

# initial guess for optimization

initial_g = number_of_stocks * [1/number_of_stocks]

optimized_sr = sci_opt.minimize(neg_sr, initial_g, method='SLSQP', bounds=bounds, constraints=constraints)

print(optimized_sr)


