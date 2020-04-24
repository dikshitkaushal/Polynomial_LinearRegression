# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:42:21 2020

@author: DELL
"""

"import the libraries"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"import the dataset"
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

"training the linear Regression model on a whole Dataset"
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

"training the polynomial Regression model on a whole Dataset"
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

"Visualising the linear Regression Results"
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title("Truth or bluff (Linear Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

"Visulaising the Polynomial Regression Results"
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title("Truth or bluff (polynomial Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"Visualising more smoother curve"
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title("Truth or bluff (polynomial Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"predicting a new result with Linear Regression"
lin_reg.predict([[6.5]])

"predicting a new result with polynomial Regression"
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))