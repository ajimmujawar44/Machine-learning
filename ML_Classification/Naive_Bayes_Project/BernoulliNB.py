# -*- coding: utf-8 -*-
"""
Bernoulli Naive Bayes Classification

Created on Wed Jul 22 2026

@author: Lenovo
"""

# ============================================================
# Import Libraries
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Load Dataset
# ============================================================

dataset = pd.read_csv("data/logistic_classification.csv")

print(dataset.head())

# ============================================================
# Independent & Dependent Variables
# ============================================================

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# ============================================================
# Train Test Split
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)

# ============================================================
# Feature Scaling
# ============================================================

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ============================================================
# Convert Continuous Features into Binary Features
# ============================================================

X_train = (X_train > 0).astype(int)
X_test = (X_test > 0).astype(int)

print("\nSample Binary Features (Training):")
print(X_train[:5])

# ============================================================
# Train Bernoulli Naive Bayes
# ============================================================

from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(X_train, y_train)

# ============================================================
# Prediction
# ============================================================

y_pred = classifier.predict(X_test)

# ============================================================
# Evaluation
# ============================================================

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", accuracy)

print("Precision :", precision_score(y_test, y_pred))

print("Recall :", recall_score(y_test, y_pred))

print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# ============================================================
# Bias & Variance
# ============================================================

bias = classifier.score(X_train, y_train)

variance = classifier.score(X_test, y_test)

print("\nTraining Accuracy (Bias):", bias)

print("Testing Accuracy (Variance):", variance)

# ============================================================
# ROC Curve
# ============================================================

from sklearn.metrics import roc_curve, roc_auc_score

y_prob = classifier.predict_proba(X_test)[:, 1]

fpr, tpr, threshold = roc_curve(y_test, y_prob)

auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8,6))

plt.plot(fpr, tpr, label="BernoulliNB (AUC = %.3f)" % auc)

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Bernoulli Naive Bayes")

plt.legend()

plt.grid()

plt.show()

# ============================================================
# Confusion Matrix Plot
# ============================================================

plt.figure(figsize=(5,5))

plt.imshow(cm, cmap="Blues")

plt.colorbar()

plt.xticks([0,1],["0","1"])

plt.yticks([0,1],["0","1"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j],
                 ha='center',
                 va='center',
                 color='black')

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ============================================================
# Accuracy Plot
# ============================================================

plt.figure(figsize=(5,5))

plt.bar(["BernoulliNB"], [accuracy])

plt.ylim(0,1)

plt.ylabel("Accuracy")

plt.title("Bernoulli Naive Bayes Accuracy")

plt.show()

# ============================================================
# Bias vs Variance Plot
# ============================================================

plt.figure(figsize=(5,5))

plt.bar(["Training Accuracy", "Testing Accuracy"],
        [bias, variance])

plt.ylim(0,1)

plt.ylabel("Accuracy")

plt.title("Bias vs Variance")

plt.show()

# ============================================================
# Prediction Example
# ============================================================

sample = np.array([[35, 60000]])

sample = sc.transform(sample)

sample = (sample > 0).astype(int)

prediction = classifier.predict(sample)

print("\nPrediction for Age=35 and Salary=60000 :", prediction[0])