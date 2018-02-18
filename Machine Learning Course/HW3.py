#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:10:33 2018

@author: Claudio
"""
## 3.1
import numpy as np
import pandas as pd
import statsmodels.api as sm 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.linear_model import LinearRegression 
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, cross_val_predict 
from sklearn import metrics
np.random.seed(6996) # error term
epsilon_vec = np.random.normal(0,1,500).reshape(500,1) # X_matrix or regressors or predictiors
X_mat = np.random.normal(0,2,size = (500,500))
# Slope
slope_vec = np.random.uniform(1,5,500)
# Simulate Ys
Y_mat = 1 + np.cumsum(X_mat * slope_vec,axis=1)[:,1:] + epsilon_vec
# each col of Y_mat representing one simulation vector: starting with 2 re gressors, end with 500
#print(Y_mat.shape)
data = pd.DataFrame(np.column_stack([X_mat[:,:1],Y_mat[:,:1]]),columns=['X_mat','Y_mat']) 

Xmat_train, Xmat_test, Ymat_train, Ymat_test = train_test_split(X_mat[:,:1], Y_mat[:,:1], test_size = 0.2, random_state = 0)

## 3.2
for i in range(2,11):
    colname = 'X_mat_%d'%i #new var will be x_power 
    data[colname] = data['X_mat']**i
 
print(data.head())
#initialize predictors:
def linear_regression(data, power):
    predictors=['X_mat']
    if power>=2:
        predictors.extend(['X_mat_%d'%i for i in range(2,power+1)])
    #Fit the model
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors],data['Y_mat'])
    y_pred = linreg.predict(data[predictors])
   
    #Return the result in pre-defined format
    rss = sum((y_pred-data['Y_mat'])**2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret
    #Initialize a dataframe to store the results:
col = ['rss','intercept'] + ['coef_X_mat_%d'%i for i in range(1,11)] 
ind = ['model_pow_%d'%i for i in range(1,11)] 
m10 = pd.DataFrame(index=ind, columns=col)
#Iterate through all powers and assimilate results 
for i in range(1,11):
    m10.iloc[i-1,0:i+2] = linear_regression(data, power=i)
pd.options.display.float_format = '{:,.2g}'.format 
m10

##question 3.3
from sklearn.linear_model import Ridge
def ridge_regression(data, predictors, alpha):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True) 
    hi = ridgereg.fit(data[predictors],data['Y_mat']) 
    y_pred = ridgereg.predict(data[predictors])
#Return the result in pre-defined format
    rss = sum((y_pred-data['Y_mat'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    #cross validation
    scores = cross_val_score(hi, data, Y_mat, cv=6)
    predx = cross_val_predict(hi, data, Y_mat, cv=6)
    accuracy = metrics.r2_score(Y_mat, predx)
    print("Cross-Predicted Accuracy:", accuracy)
    print("Cross-validated scores:", scores)
    return ret
#Initialize predictors to be set of 15 powers of x 
predictors=['Y_mat']
predictors.extend(['X_mat_%d'%i for i in range(2,11)])
#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_X_mat_%d'%i for i in range(1,11)] 
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)] 
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)
for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i])
#Set the display format to be scientific for ease of analysis 
pd.options.display.float_format = '{:,.2g}'.format 
coef_matrix_ridge
coef_matrix_ridge.apply(lambda x: sum(x.values==0),axis=1)

##question 3.4
from sklearn.linear_model import Lasso
def lasso_regression(data, predictors, alpha):
#Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data['Y_mat'])
    y_pred = lassoreg.predict(data[predictors])
#Return the result in pre-defined format
    rss = sum((y_pred-data['Y_mat'])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret
#Initialize predictors to all 15 powers of x
predictors=['X_mat']
predictors.extend(['X_mat_%d'%i for i in range(2,11)])
#Define the alpha values to test
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]
#Initialize the dataframe to store coefficients
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,11)] 
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)
#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(data, predictors, alpha_lasso[i])

pd.options.display.float_format = '{:,.2g}'.format 
coef_matrix_lasso
coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)

