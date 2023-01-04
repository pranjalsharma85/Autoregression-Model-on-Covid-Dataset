import pandas as pd 
import matplotlib.pyplot as plt 
import operator
import numpy as np 
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.ar_model import AutoReg as AR
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
train_size = 0.65 # 35% for testing
X = series.values

train, test = X[:int(len(X)*train_size)], X[int(len(X)*train_size):]

def ARmodel(train_data, test_data, lag):
    window=lag
    model = AR(train_data, lags=window)
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model

    #using these coefficients walk forward over time steps in test, one step each time
    history = train_data[len(train_data)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test_data)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1] # Add other values
        obs = test_data[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual test value to history, to be used in next step.'''
    rmse_=mean_squared_error(test_data, predictions,squared=False)*100/(sum(test_data)/len(test_data))
    mape_=mean_absolute_percentage_error(test_data, predictions)
    return rmse_, mape_

lag=[1,5,10,15,25]
rmse_list=[]
mape_list=[]

for i in lag:
    rmse, mape=ARmodel(train, test,i)
    rmse_list.append(rmse[0])
    mape_list.append(mape)
print(rmse_list, mape_list)
plt.bar(lag, rmse_list)
plt.ylabel('RMSE error')
plt.xlabel('Lag values')
plt.title("Q3\n Bar chart between RMSE and Lag values")
plt.xticks(lag)
plt.show()

plt.bar(lag, mape_list)
plt.ylabel('MAPE error')
plt.xlabel('Lag values')
plt.title("Q3\n Bar chart between MAPE and Lag values")
plt.xticks(lag)
plt.show()
