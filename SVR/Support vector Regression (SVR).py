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


# pol model pred
poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)

# simple lin model pred
lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)




from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# ---------------- Default SVR ----------------
svr_regressor = SVR()          # Default: kernel='rbf', degree=3, gamma='scale'

svr_regressor.fit(X, y)

svr_model_pred = svr_regressor.predict([[6.5]])
print("Default SVR Prediction :", svr_model_pred)


# ---------------- RBF Kernel ----------------
svr_regressor = SVR(kernel='rbf', degree=3, gamma='scale')

svr_regressor.fit(X, y)

svr_model_pred = svr_regressor.predict([[6.5]])
print("RBF Kernel Prediction :", svr_model_pred)


# ---------------- Sigmoid Kernel ----------------
svr_regressor = SVR(kernel='sigmoid', degree=3, gamma='scale')

svr_regressor.fit(X, y)

svr_model_pred = svr_regressor.predict([[6.5]])
print("Sigmoid Kernel Prediction :", svr_model_pred)


# ---------------- Polynomial Degree = 4 ----------------
svr_regressor = SVR(kernel='poly', degree=4, gamma='scale')

svr_regressor.fit(X, y)

svr_model_pred = svr_regressor.predict([[6.5]])
print("Polynomial (Degree=4, Gamma=scale) :", svr_model_pred)


# ---------------- Polynomial Degree = 4, Gamma = auto ----------------
svr_regressor = SVR(kernel='poly', degree=4, gamma='auto')

svr_regressor.fit(X, y)

svr_model_pred = svr_regressor.predict([[6.5]])
print("Polynomial (Degree=4, Gamma=auto) :", svr_model_pred)


# ---------------- Polynomial Degree = 5 ----------------
svr_regressor = SVR(kernel='poly', degree=5, gamma='scale')

svr_regressor.fit(X, y)

svr_model_pred = svr_regressor.predict([[6.5]])
print("Polynomial (Degree=5, Gamma=scale) :", svr_model_pred)


#  Basic SVR Visualization 

import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))

# Actual data
plt.scatter(X,y, color = 'red', s = 80, label = ' Actual Data')

# SVR Prediction line 

plt.plot(X, svr_regressor.predict(X), color='blue',linewidth=3, label='SVR Model')

plt.title("Support Vector Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.grid(True)

plt.show()

# Smooth SVR Curve 

# SVR look much better with more X values

X_grid = np.arange(X.min(), X.max(), 0.01)
X_grid = X_grid.reshape(-1,1)

plt.figure(figsize=(8,5))

plt.scatter(X,y, color='red', s=70, label='Actual Data')

plt.plot(X_grid, svr_regressor.predict(X_grid),color='gray',linewidth=3, label= 'SVR Curve')

plt.title("Smooth support Vector Regression")
plt.xlabel("Position level")
plt.ylabel("Salary")

plt.legend()
plt.grid(True)

plt.show()


# Compare Linear vs Plynomial vs SVR

plt.figure (figsize=(10,6))

# Actual data
plt.scatter(X,y, color='purple',s=80, label='Actual Data')

# Linear regression

plt.plot(X, lin_reg.predict(X), color='red',linewidth=2, label='Linear Rehression')

# polynomial Regression

plt.plot(X_grid, lin_reg_2.predict(poly_reg.transform(X_grid)),color='green',linewidth=3,label='Polynomial regression')

# SVR 

plt.plot(X_grid, svr_regressor.predict(X_grid),color='gold',linewidth=3,label='SVR')

plt.title("Linear vs Polynomial vs SVR")

plt.xlabel("Positional level")

plt.ylabel("Salary")

plt.legend()

plt.grid(True)

plt.show()

# Actual vs Predicted Scatter

prediction = svr_regressor.predict(X)

plt.figure(figsize=(6,5))

plt.scatter(y,
            prediction,
            color='purple')

plt.xlabel("Actual Salary")

plt.ylabel("Predicted Salary")

plt.title("Actual vs Predicted (SVR)")

plt.grid(True)

plt.show()

# Residual Error Visualization

prediction = svr_regressor.predict(X)

plt.figure(figsize=(8,5))

plt.scatter(X,
            y,
            color='red')

plt.plot(X,
         prediction,
         color='green')

for i in range(len(X)):
    plt.vlines(X[i],
               y[i],
               prediction[i],
               color='blue',
               linestyle='dashed')

plt.title("Residual Errors in SVR")

plt.xlabel("Position Level")

plt.ylabel("Salary")

plt.show()

# Professional SVR Graph

plt.figure(figsize=(9,6))

plt.scatter(X,
            y,
            color='red',
            s=120,
            edgecolors='black',
            label='Employee Salary')

plt.plot(X_grid,
         svr_regressor.predict(X_grid),
         color='darkgreen',
         linewidth=4,
         label='Support Vector Regression')

plt.xlabel("Position Level", fontsize=12)

plt.ylabel("Salary", fontsize=12)

plt.title("Employee Salary Prediction using SVR",
          fontsize=14)

plt.grid(True)

plt.legend()

plt.show()




















