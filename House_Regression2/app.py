from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
print("Running file:", os.path.abspath(__file__))
print("=" * 50)
print("THIS IS THE APP.PY I AM RUNNING")
print("=" * 50)

app = Flask(__name__)

# --------------------------
# Project Paths
# --------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = BASE_DIR
# --------------------------
# Load Models
# --------------------------

model_names = [
    "LinearRegression",
    "HuberRegression",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNet",
    "PolynomialRegression",
    "SGDRegression",
    "ANN",
    "RandomForest",
    "SVM",
    "LGBM",
    "XGBoost",
    "KNN"
]

models = {}

for model_name in model_names:
    file_path = os.path.join(MODEL_PATH, model_name + ".pkl")

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            models[model_name] = pickle.load(f)
        print(f"Loaded: {model_name}")
    else:
        print(f"Missing: {file_path}")

# --------------------------
# Load Evaluation Results
# --------------------------

csv_path = os.path.join(MODEL_PATH, "model_evaluation_results.csv")

if os.path.exists(csv_path):
    results_df = pd.read_csv(csv_path)
else:
    results_df = pd.DataFrame()

# --------------------------
# Home
# --------------------------

@app.route("/")
def home():

    if not results_df.empty:
        best_model = results_df.sort_values(
            by="R2 Score",
            ascending=False
        ).iloc[0]["Model"]
    else:
        best_model = "Not Available"

    return render_template(
        "index.html",
        models=model_names,
        best_model=best_model
    )
# --------------------------
# Prediction
# --------------------------

@app.route("/predict", methods=["POST"])
def predict():

    model_name = request.form.get("model")

    if model_name not in model_names:
        return "Selected model not found.", 404

    try:

        features = pd.DataFrame([{

            "Avg. Area Income":
            float(request.form.get("income")),

            "Avg. Area House Age":
            float(request.form.get("age")),

            "Avg. Area Number of Rooms":
            float(request.form.get("rooms")),

            "Avg. Area Number of Bedrooms":
            float(request.form.get("bedrooms")),

            "Area Population":
            float(request.form.get("population"))

        }])

        prediction = models[model_name].predict(features)[0]

        prediction = round(float(prediction), 2)

        return render_template(
            "results.html",
            prediction=prediction,
            model_name=model_name
        )

    except Exception as e:
        return f"Prediction Error: {e}"

# --------------------------
# Leaderboard
# --------------------------

#@app.route("/leaderboard")
#def leaderboard():
#    return "<h1>Leaderboard Route Working</h1>"
# --------------------------
# About
# --------------------------
@app.route("/leaderboard")
def leaderboard():
    return render_template(
    "leaderboard.html",
    table=results_df.to_html(
        classes="table table-striped table-bordered",
        index=False
    )
)

    

@app.route("/about")
def about():
    return "<h2>House Price Prediction Project</h2>"

# Print all registered routes
print("\nAvailable Routes:")
for rule in app.url_map.iter_rules():
    print(rule)

if __name__ == "__main__":
    app.run(debug=True)