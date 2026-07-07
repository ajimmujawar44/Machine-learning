# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 22:15:12 2026

@author: Lenovo
"""

# House Price Prediction using Backward Elimination

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import Dataset
df = pd.read_csv(r"C:\Users\Lenovo\Downloads\House_data.csv")

# Display first 5 rows
print(df.head())

# Check Missing Values
print(df.isnull().any())

# Check Data Types
print(df.dtypes)

# Drop unnecessary columns
df = df.drop(['id', 'date'], axis=1)

# Pair Plot
with sns.plotting_context("notebook", font_scale=1.5):
    g = sns.pairplot(
        df[['sqft_lot',
            'sqft_above',
            'price',
            'sqft_living',
            'bedrooms']],
        hue='bedrooms',
        palette='tab20',
        height=3
    )

g.set(xticklabels=[])

plt.show()

# Independent and Dependent Variables

x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Train Test Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=1/3,
    random_state=0
)

# Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Prediction

y_pred = regressor.predict(x_test)

# ==========================================================
# Backward Elimination (Step by Step)
# ==========================================================

import statsmodels.api as sm

# Add Constant

x_opt = sm.add_constant(x)

# Initial OLS Model

regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()

print(regressor_OLS.summary())

# ==========================================================
# Backward Elimination Function
# ==========================================================

def backwardElimination(x, SL):

    numVars = x.shape[1]

    temp = np.zeros((x.shape[0], x.shape[1]))

    for i in range(numVars):

        regressor_OLS = sm.OLS(y, x).fit()

        maxVar = max(regressor_OLS.pvalues)

        adjR_before = regressor_OLS.rsquared_adj

        if maxVar > SL:

            for j in range(x.shape[1]):

                if regressor_OLS.pvalues[j] == maxVar:

                    temp[:, j] = x[:, j]

                    x = np.delete(x, j, axis=1)

                    tmp_regressor = sm.OLS(y, x).fit()

                    adjR_after = tmp_regressor.rsquared_adj

                    if adjR_before >= adjR_after:

                        x = np.insert(x, j, temp[:, j], axis=1)

                        print("\nAdjusted R² decreased.")
                        print("Rolling back deleted variable.\n")

                        print(regressor_OLS.summary())

                        return x

                    else:

                        break

    print("\nFinal Model\n")

    print(regressor_OLS.summary())

    return x

# Significance Level

SL = 0.05

# Add Constant Again

x_opt = sm.add_constant(x)

# Final Selected Variables

x_Modeled = backwardElimination(x_opt, SL)

print("\nFinal Shape of Dataset :", x_Modeled.shape)