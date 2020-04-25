# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:47:15 2020

@author: DELL
"""

#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

#fitting the random forest regression into the model
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x,y)

#visualising the result
x_label=np.arange(min(x),max(x),0.01)
x_label=x_label.reshape(len(x_label),1)
plt.scatter(x,y,color='red')
plt.plot(x_label,regressor.predict(x_label),color='blue')
plt.title('Truth or bluff (Random Forest)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predict a new R3sult
y_predict=regressor.predict([[6.5]])