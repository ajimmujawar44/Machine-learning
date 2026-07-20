# Logistic Regression Classification

A simple Machine Learning project to understand and implement the **Logistic Regression Classification** algorithm using Python and Scikit-learn.

## Dataset

- Social_Network_Ads.csv

## Features

- Age
- Estimated Salary

## Target

- Purchased (0 = No, 1 = Yes)

## Project Workflow

- Import libraries
- Load dataset
- Split training and testing data
- Feature scaling
- Train Logistic Regression model
- Predict test data
- Evaluate model
  - Confusion Matrix
  - Accuracy Score
  - Classification Report
  - ROC Curve
- Visualize Training Set
- Visualize Test Set
- Future Prediction (Optional)

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure

```
Logistic_Regression_Project/
│
├── data/
│   └── Social_Network_Ads.csv
│
├── screenshots/
│   ├── training_set.png
│   ├── test_set.png
│   ├── roc_curve.png
│   └── output.png
│
├── logistic_regression.py
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

```bash
pip install -r requirements.txt
python logistic_regression.py
```

## Output

The project generates:

- Confusion Matrix
- Accuracy Score
- Classification Report
- ROC Curve
- Training Set Visualization
- Test Set Visualization