import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Print complete arrays (optional)
np.set_printoptions(threshold=sys.maxsize)

# Load Dataset
dataset = pd.read_csv(r"C:\Users\Lenovo\Downloads\House_data.csv")

# Independent and Dependent variables
x = dataset[['sqft_living']].values
y = dataset['price'].values

# Split the data into Train and Test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.33,
    random_state=0
)

# Train the model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict
y_pred = regressor.predict(x_test)

# ==========================
# Training Set Graph
# ==========================
plt.figure(figsize=(8,5))

plt.scatter(x_train, y_train, color='red', label='Training Data')

# Sort x values before plotting
sort_train = np.argsort(x_train[:, 0])

plt.plot(
    x_train[sort_train],
    regressor.predict(x_train)[sort_train],
    color='blue',
    linewidth=2,
    label='Regression Line'
)

plt.title("Training Set")
plt.xlabel("Living Area (sqft)")
plt.ylabel("Price")
plt.legend()
plt.show()

# ==========================
# Test Set Graph
# ==========================
plt.figure(figsize=(8,5))

plt.scatter(x_test, y_test, color='green', label='Test Data')

sort_test = np.argsort(x_test[:, 0])

plt.plot(
    x_test[sort_test],
    regressor.predict(x_test)[sort_test],
    color='blue',
    linewidth=2,
    label='Regression Line'
)

plt.title("Test Set")
plt.xlabel("Living Area (sqft)")
plt.ylabel("Price")
plt.legend()
plt.show()

import pickle
filename ='House_Prediction_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("model has been pickled and saved as House_Prediction_model_.pkl")
import os
print(os.getcwd())