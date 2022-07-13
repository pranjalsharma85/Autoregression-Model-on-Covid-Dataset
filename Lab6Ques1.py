#Pranjal Sharma
#B20305
#7374065064
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

readfile=pd.read_csv("C:\\DS3_lab_6\\daily_covid_cases.csv")
original=readfile['new_cases']

# Part A
figure, axis = plt.subplots(figsize=(10,6))
axis.plot(readfile['Date'],
           readfile['new_cases'].values,
           color='red')
axis.set(xlabel="Date", ylabel="new_cases",
       title="Part A")
dateformat = DateFormatter("%b-%d")
axis.xaxis.set_major_formatter(dateformat)
axis.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation =45)
plt.show()

# Part B
lag_1=original.shift(1)
print(" Autocorrelation (Pearson correlation coefficient) :-",original.corr(lag_1))
print()

# Part C
plt.scatter(original, lag_1, s=5)
plt.xlabel("Given Data")
plt.ylabel("Lag-1 time series Data")
plt.title("Part C")
plt.show()

# Part D
PCC=sm.tsa.acf(original)
Lag_value=[1,2,3,4,5,6]
pcc=PCC[1:7]
plt.plot(Lag_value,pcc, marker='o')
for xitem,yitem in np.nditer([Lag_value, pcc]):
        etiqueta = "{:.3f}".format(yitem)
        plt.annotate(etiqueta, (xitem,yitem), textcoords="offset points",xytext=(0,10),ha="center")
plt.xlabel("Lag value")
plt.ylabel("Correlation coefficient value")
plt.title("Part D")
plt.show()

# Part E
plot_acf(x=original, lags=50)
plt.xlabel("Lag value")
plt.ylabel("Correlation coefficient value")
plt.title("Part E")
plt.show()