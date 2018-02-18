#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:53:08 2018

@author: Claudio
"""

#simple linear regression
# importing libraries
import numpy as np #to manage arrays and matrices and high lvl math functions
import matplotlib.pyplot as plt #creates figures and plots
import pandas as pd #help manage datasets

# importing data set
dataset = pd.read_csv('Salary_Data.csv')
#matrix of features
X = dataset.iloc[:,:-1].values #independent variables
y = dataset.iloc[:,1].values #dependent variables
print(X)
print(y)

#splitting the data set intro training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train) #this makes the machine learn the correlation between
#the x and y variables and finds a linear regression for them

#predicting the test set results
y_pred = regressor.predict(X_test)

#visualizing the training results

"""plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color ='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()"""

#visualizing the test results
#we first train the model with the training set (the regression) and then we verify it
#with the test set to see if the model was actually trained and followed
#the numbers of the test result meaning that it was a good predictive model
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color ='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()