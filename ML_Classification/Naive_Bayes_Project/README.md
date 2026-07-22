# Naive Bayes Classification Project

## Overview

This project demonstrates different types of Naive Bayes Classification algorithms using the Social Network Ads dataset (`logistic_classification.csv`).

The objective is to understand how each Naive Bayes variant works, compare their performance, and learn when each algorithm should be used.

---

## Algorithms Implemented

- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Naive Bayes Class Work Example

---

## Dataset

Dataset Name

```
logistic_classification.csv
```

Input Features

- Age
- Estimated Salary

Target Variable

- Purchased (0 = No, 1 = Yes)

---

## Project Structure

```
Naive_Bayes_Classification_Project/
│
├── data/
│   └── logistic_classification.csv
│
├── 01_Gaussian_Naive_Bayes.py
├── 02_Multinomial_Naive_Bayes.py
├── 03_Bernoulli_Naive_Bayes.py
├── 04_class_work_navie_bayes.py
│
├── Screenshots/
├── README.md
└── requirements.txt
```

---

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Topics Covered

- Data Loading
- Train-Test Split
- Feature Scaling
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Prediction
- Confusion Matrix
- Accuracy Score
- Precision
- Recall
- F1 Score
- Classification Report
- ROC Curve
- AUC Score
- Bias
- Variance
- Visualization

---

## Results

The project compares different Naive Bayes algorithms on the same dataset.

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Bias
- Variance

---

## Note

Gaussian Naive Bayes is naturally suitable for continuous numerical features.

Multinomial Naive Bayes and Bernoulli Naive Bayes are demonstrated for educational purposes. They are generally intended for count-based or binary feature datasets such as text classification.

---

## How to Run

Clone Repository

```bash
git clone https://github.com/ajimmujawar44/Machine-learning.git
```

Move into Project

```bash
cd ML_Classification/Naive_Bayes_Classification_Project
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python 01_Gaussian_Naive_Bayes.py
```

---

