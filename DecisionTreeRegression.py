# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 00:08:15 2020

@author: DELL
"""

#import the libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#import the dataset
dataset= pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

#train the model using decision tree
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#visulaise the more resoluton result 
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Truth Or Bluff (decision Tree)')
plt.xlabel('position level ')
plt.ylabel('salaries')
plt.show()