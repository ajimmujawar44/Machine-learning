import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load Dataset
data = pd.read_csv(r"E:\ML\Machine learning\House_Regression\USA_Housing.csv")

# Features and Target
X = data.drop(["Price", "Address"], axis=1)
y = data["Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Polynomial Regression Model
poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression())
])

# Train
poly_model.fit(X_train, y_train)

# Save Model
with open("PolynomialRegression.pkl", "wb") as f:
    pickle.dump(poly_model, f)

print("PolynomialRegression.pkl created successfully!")