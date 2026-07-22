# -*- coding: utf-8 -*-
"""
Gaussian Naive Bayes Classification

Created on Wed Jul 22 2026

@author: Lenovo
"""

# ============================================================
#               Importing the Libraries
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
#               Importing Dataset
# ============================================================

dataset = pd.read_csv(r"data/logistic_classification.csv")
print(dataset.head())

# ============================================================
#          Independent & Dependent Variables
# ============================================================

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

# ============================================================
#             Splitting Dataset
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)

#########################################################################
###################### WITHOUT FEATURE SCALING ##########################
#########################################################################

print("\n================ WITHOUT FEATURE SCALING ================\n")

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# ============================================================
#               Evaluation
# ============================================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix")
print(cm)

accuracy1 = accuracy_score(y_test, y_pred)
print("\nAccuracy :", accuracy1)

print("\nPrecision :", precision_score(y_test,y_pred))

print("Recall :", recall_score(y_test,y_pred))

print("F1 Score :", f1_score(y_test,y_pred))

print("\nClassification Report\n")

print(classification_report(y_test,y_pred))

# ============================================================
#           Bias & Variance
# ============================================================

bias1 = classifier.score(X_train,y_train)

variance1 = classifier.score(X_test,y_test)

print("\nTraining Accuracy (Bias):",bias1)

print("Testing Accuracy (Variance):",variance1)

#########################################################################
######################### FEATURE SCALING ###############################
#########################################################################

print("\n================ WITH FEATURE SCALING ================\n")

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)

X_test_sc = sc.transform(X_test)

classifier2 = GaussianNB()

classifier2.fit(X_train_sc,y_train)

y_pred2 = classifier2.predict(X_test_sc)

# ============================================================
#               Evaluation
# ============================================================

cm2 = confusion_matrix(y_test,y_pred2)

print("Confusion Matrix")

print(cm2)

accuracy2 = accuracy_score(y_test,y_pred2)

print("\nAccuracy :",accuracy2)

print("\nPrecision :",precision_score(y_test,y_pred2))

print("Recall :",recall_score(y_test,y_pred2))

print("F1 Score :",f1_score(y_test,y_pred2))

print("\nClassification Report\n")

print(classification_report(y_test,y_pred2))

bias2 = classifier2.score(X_train_sc,y_train)

variance2 = classifier2.score(X_test_sc,y_test)

print("\nTraining Accuracy (Bias):",bias2)

print("Testing Accuracy (Variance):",variance2)

#########################################################################
############################ ROC CURVE ##################################
#########################################################################

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

y_prob = classifier2.predict_proba(X_test_sc)[:,1]

fpr,tpr,threshold = roc_curve(y_test,y_prob)

auc = roc_auc_score(y_test,y_prob)

plt.figure(figsize=(8,6))

plt.plot(fpr,tpr,label="GaussianNB AUC = %.3f"%auc)

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

#########################################################################
###################### ACCURACY COMPARISON ##############################
#########################################################################

plt.figure(figsize=(6,5))

models=["Without Scaling","With Scaling"]

accuracy=[accuracy1,accuracy2]

plt.bar(models,accuracy)

plt.ylabel("Accuracy")

plt.title("Gaussian NB Accuracy Comparison")

plt.ylim(0,1)

plt.show()

#########################################################################
###################### BIAS VARIANCE GRAPH ##############################
#########################################################################

plt.figure(figsize=(6,5))

train=[bias1,bias2]

test=[variance1,variance2]

x=np.arange(2)

plt.bar(x-0.2,train,width=0.4,label="Train")

plt.bar(x+0.2,test,width=0.4,label="Test")

plt.xticks(x,["Without Scaling","With Scaling"])

plt.ylabel("Accuracy")

plt.title("Bias Variance Comparison")

plt.legend()

plt.show()