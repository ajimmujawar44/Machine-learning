# import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv(r"C:\Users\Lenovo\Downloads\emp_sal.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.figure(figsize=(6,4))
plt.scatter(x,y, color ='red', label = 'Actual data')
plt.plot(x, lin_reg.predict(x),color = 'blue',linewidth=3, label='Regression line')
plt.title("Linear Regression")

plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Polynominal Regression

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=4)

x_poly = poly.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.figure(figsize=(6,4))
plt.scatter(x,y, color='blue')
plt.plot(x, lin_reg2.predict(x_poly), color ='green', linewidth =3)

plt.title("Polynominal regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")

plt.show()

# Polynominal Smooth Graph
#  here grid create many ponts our datast contain only 10 points, (more points= smooth curve)
x_grid = np.arange(x.min(),x.max(),0.1)
x_grid = x_grid.reshape(-1,1)

plt.figure(figsize=(6,4))
plt.scatter(x,y, color = 'red')
plt.plot(x_grid, lin_reg2.predict(poly.transform(x_grid)),color = 'green', linewidth =3)
plt.title("Smooth Polynomial Curve")
plt.xlabel("Position Level")
plt.ylabel("Salary")

plt.show()
