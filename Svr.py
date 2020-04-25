# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:26:51 2020

@author: DELL
"""

#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(np.reshape(y,(10,1)))

#fitting the svr model into the dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

#visualising the SVR results
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title("Truth Or Bluff (svr)")
plt.xlabel("Position Level")
plt.ylabel("salary")
plt.show()

#predicting a new result
"""HERE THE ANSWER WOULD COME IN SCALAR FORMAT SO TO AVOID THIS WE 
    NEED TO PERFORM INVERSE TRANSFORM """
    
"""AND WE CAN'T ENTER THE VALUE DIRECTLT SO WE NEED TO SCALE IT FIRST"""
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))