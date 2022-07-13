#Pranjal Sharma
#B20305
#7374065064
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.ar_model import AutoReg as AR
readfile = pd.read_csv("C:\\Users\\pranj\\Desktop\\daily_covid_cases.csv",parse_dates=['Date'],index_col=['Date'],sep=',')
TrainData_size = 0.65 # we splitted the given data into 65:35 ratio of train:test data. 
X = readfile.values
train, test = X[:int(len(X)*TrainData_size)], X[int(len(X)*TrainData_size):]

def ARmodel(train_data, test_data, lag):
    window=lag
    model = AR(train_data, lags=window)
    modelfit = model.fit() # fitting or training of the model
    coefficients = modelfit.params # Getting the coefficients of AR model


    prevdata = train_data[len(train_data)-window:]
    prevdata = [prevdata[i] for i in range(len(prevdata))]
    predicted_data = list() 
    for t in range(len(test_data)):
        length = len(prevdata)
        lag = [prevdata[i] 
        for i in range(length-window,length)]
        ycap = coefficients[0] # Initializing to w0
        for d in range(window):
            ycap += coefficients[d+1] * lag[window-d-1] # Adding other values
        observation = test_data[t]
        predicted_data.append(ycap) 
    RMSE=mean_squared_error(test_data, predicted_data,squared=False)*100/(sum(test_data)/len(test_data))
    MAPE=mean_absolute_percentage_error(test_data, predicted_data)
    return RMSE, MAPE

lag=[1,5,10,15,25]
RMSElst=[]
MAPElst=[]
for i in lag:
    RMSE, MAPE=ARmodel(train, test,i)
    RMSElst.append(RMSE[i])
    MAPElst.append(MAPE[i])

plt.bar(lag, RMSElst)
plt.ylabel('RMSE')
plt.xlabel('Lag values')
plt.title("Bar chart between RMSE and Lag values")
plt.xticks(lag)
plt.show()

plt.bar(lag, MAPElst)
plt.ylabel('MAPE')
plt.xlabel('Lag values')
plt.title("Bar chart between MAPE and Lag values")
plt.xticks(lag)
plt.show()
