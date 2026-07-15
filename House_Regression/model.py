import pandas as pd
import pickle

# Train-Test Split
from sklearn.model_selection import train_test_split

# Regression Models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
    HuberRegressor
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

import lightgbm as lgb
import xgboost as xgb

# Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

data = pd.read_csv(r"E:\ML\Machine learning\House_Regression\USA_Housing.csv")

# -------------------------------------------------------
# Data Preprocessing
# -------------------------------------------------------

X = data.drop(['Price', 'Address'], axis=1)   # Use 'price' if your column is lowercase
y = data['Price']

# -------------------------------------------------------
# Train-Test Split
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

# -------------------------------------------------------
# Define Models
# -------------------------------------------------------

models = {
    "LinearRegression": LinearRegression(),

    "HuberRegression": HuberRegressor(),

    "RidgeRegression": Ridge(),

    "LassoRegression": Lasso(),

    "ElasticNet": ElasticNet(),

    "PolynomialRegression": Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("linear", LinearRegression())
    ]),

    "SGDRegression": SGDRegressor(
        max_iter=1000,
        tol=1e-3,
        random_state=0
    ),

    "ANN": MLPRegressor(
        hidden_layer_sizes=(100,),
        max_iter=1000,
        random_state=0
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        random_state=0
    ),

    "SVM": SVR(),

    "LGBM": lgb.LGBMRegressor(
        random_state=0
    ),

    "XGBoost": xgb.XGBRegressor(
        random_state=0,
        objective='reg:squarederror'
    ),

    "KNN": KNeighborsRegressor()
}

# -------------------------------------------------------
# Train Models & Evaluate
# -------------------------------------------------------

results = []

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    # Save Model
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# -------------------------------------------------------
# Save Results
# -------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

results_df.to_csv(
    "model_evaluation_results.csv",
    index=False
)

print("\nAll models have been trained successfully.\n")

print(results_df)