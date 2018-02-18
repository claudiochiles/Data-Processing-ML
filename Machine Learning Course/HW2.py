#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:00:14 2018

@author: Claudio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from scipy import stats
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from numpy.linalg import inv

data = pd.read_csv('session_2_homework_data.csv')

##question 1
X = data.iloc[:,1:8]
y = data.iloc[:,8:9]
#graph1 = plt.plot(X)
#plt.show(graph1)



##question 2 tables
def lin_reg(X,y):
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    return [reg.intercept_[0],reg.coef_[0][0]]

t1 = []
for column in X:
    t1.append(lin_reg(data[column].to_frame(),y.Output1.to_frame()))
table1 = pd.DataFrame(data = t1)
table1.columns = ['Intercept','Slope']

t2 = []
for column in X:
    t2.append(lin_reg(y.Output1.to_frame(),data[column].to_frame()))
table2 = pd.DataFrame(data = t2)
table2.columns = ['Intercept','Slope']

## question 3
z = data.iloc[:,9:11]
data['Binary'] = z.max(axis=1)
conditions = [(data['Easing'] == 1) ,(data['Tightening']== 1)]
choices = [0,1]
data['Binary'] = np.select(conditions,choices,default='nan')
data2 = data.loc[data['Binary'] != 'nan']
data2.index = range(len(data2.index))
binary = data2.iloc[:,11:12]
output = data2[['USGG3M','USGG3M','USGG2YR','USGG3YR','USGG5YR','USGG10YR','USGG30YR','Output1']]
binary2 = binary.apply(pd.to_numeric, errors='ignore')
plt.plot(data2.iloc[:,1:8])
plt.plot(data2.iloc[:,8:9],'r--')
plt.plot(20*binary2)
plt.show()
#logreg
M2 = data2.iloc[:,1:8]
y2 = np.ravel(binary2)
M3 = data2.iloc[:,1:2]
print('Logistic Regression Prob')
logreg = linear_model.LogisticRegression()
logreg.fit(M2,y2)
prex = logreg.predict_proba(M2)
print(prex)

logreg2 = linear_model.LogisticRegression()
logreg2.fit(M3,y2)
prex2 = logreg2.predict_proba(M3)
print(prex2)

##question 4
standardisedX = scale(X)
pca = PCA().fit(standardisedX)
##Task 1
pd.tools.plotting.scatter_matrix(X[['USGG3M','USGG2YR','USGG5YR']], diagonal="kde")
plt.tight_layout()
plt.show()
##Task 2
cov = X.cov()
#print(cov)
##Task 3
eig_vals, eig_vecs = np.linalg.eig(cov)
#print('Eigenvectors \n%s' %eig_vecs)
#print('\nEigenvalues \n%s' %eig_vals)
eig_vals2 = (eig_vals/sum(eig_vals))
objects = ('F1','F2','F3','F4','F5','F6','F7')
y_pos = np.arange(len(objects))
plt.bar(y_pos,eig_vals2,align = 'center',alpha = 0.5)
plt.xticks(y_pos, objects)
plt.show()

# Make a list of (eigenvalue, eigenvector) tuples
"""eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
#print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1), 
                      eig_pairs[1][1].reshape(7,1),
                       eig_pairs[2][1].reshape(7,1)))"""


f3loadings = eig_vecs[:,:3]
maturities = [0.25,0.5,2,3,5,10,30]
plt.figure()
plt.plot(maturities,f3loadings) 
plt.xticks([0,1,2,5,10,30])
plt.xlabel('Maturities')
plt.ylabel('Loadings')
plt.title('Importance of Factors')
plt.show()

##Task 5
def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1,len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop",
"Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)
    if out:
        print("Importance of components:",summary)
        
    return summary

#pca_summary(pca, standardisedX)
np.sum(pca.components_[0]**2)
loadings = pca.components_
load = loadings[:,0:3]


#print(loadings)
