# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:45:56 2026

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
print(sys.executable)

dataset = pd.read_csv(r"C:\Users\Lenovo\Downloads\emp_sal.csv")

print(dataset.head())

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression

print("Scikit-learn installed successfully!")



from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, y)

plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'green')
plt.title('Linear Regression graph')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

lin_model_pred  = lin_reg.predict([[6.5]])
lin_model_pred



# Polynomminal model

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()

X_poly = poly.fit_transform(X)




from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(6)

X_poly =poly_reg.fit_transform(X)




lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

print(lin_reg)   # lin reg with 1 degree
print(poly_reg)   # poly with 2 degree
print(lin_reg_2)  # lin model with 2 degree


plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'green')
plt.title('truth or bluff ("polynominal Regression")')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)
















