#Pranjal Sharma
#B20305
#7374065064
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import operator
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg as AR
readfile = pd.read_csv("C:\\DS3_lab_6\\daily_covid_cases.csv",parse_dates=['Date'],index_col=['Date'],sep=',')
TrainData_size = 0.65 # we splitted the given data into 65:35 ratio of train:test data. 
X = readfile.values
train, test = X[:int(len(X)*TrainData_size)], X[int(len(X)*TrainData_size):]

window = 5 
model = AR(train, lags=window)
modelfit = model.fit() # fitting or training of the model
coefficients = modelfit.params # Getting the coefficients of AR model
print()
print("Values of Coefficients are :",coefficients)
print()

prevdata = train[len(train)-window:]
prevdata = [prevdata[i] for i in range(len(prevdata))]
predicted_data = list() 
for t in range(len(test)):

    length = len(prevdata)

    lag = [prevdata[i] for i in range(length-window,length)]

    ycap = coefficients[0] # Initializing to w0
    
    for d in range(window):
        
        ycap += coefficients[d+1] * lag[window-d-1] # Adding other values
    observation = test[t]
    predicted_data.append(ycap) 
    prevdata.append(observation) 

# Part B (i)
plt.scatter(test,predicted_data )
plt.xlabel('Actual cases')
plt.ylabel('Predicted cases')
plt.title('Part B (i)')
plt.show()

# Part B (ii)
x=[i for i in range(len(test))]
plt.plot(x,test, label='Actual cases')
plt.plot(x,predicted_data , label='Predicted cases')
plt.legend()
plt.title('Part B (ii)')
plt.show()

# Part B (iii)
RMSE=mean_squared_error(test, predicted_data,squared=False)
print("RMSE(in %) :",RMSE*100/(sum(test)/len(test)),"%")
print()

MAPE=mean_absolute_percentage_error(test, predicted_data)
print("MAPE(in %) :",MAPE)
