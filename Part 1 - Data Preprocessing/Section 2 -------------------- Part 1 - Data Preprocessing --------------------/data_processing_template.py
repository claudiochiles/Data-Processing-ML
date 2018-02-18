#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd #help manage datasets

# importing data set
dataset = pd.read_csv('Data.csv')
#matrix of features
X = dataset.iloc[:,:-1] #dependent variables
y = dataset.iloc[:,3]#independent variables
print(X)
print(y)


#splitting the data set intro training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#scaling your data (Featuring scaling)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
#scaling your dummy variables #(dividing 
#countries into 0,0,1 doesnt really matter) and it will not affect your scaling model
#for y variable we do not have to scale it because it is a small variable

#data processing template is everything without the comments, they are put at the beginning
#of every machine learning model of this course

#missing data replacing them with the mean of the data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean',axis = 0)
imputer = imputer.fit(X[:, 1:3]) #upper bound of the columns is not included thats why you have to choose 1,3
X[:,1:3] = imputer.transform(X[:,1:3])
np.set_printoptions(threshold=100)

#Encoding categorical data (ctrl enter)
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#purchase category is done this way because it is the dependent variable
labelencoder_Y = LabelEncoder()
y= labelencoder_Y.fit_transform(y)"""