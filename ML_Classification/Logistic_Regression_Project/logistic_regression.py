# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:31:02 2026

@author: Lenovo
"""
# ---------------------------------------------
# Import Libraries
# ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------
# Import Dataset
# ---------------------------------------------
dataset = pd.read_csv(r"data/logistic_classification.csv")

# Independent Variables
X = dataset.iloc[:, [2, 3]].values

# Dependent Variable
y = dataset.iloc[:, -1].values

# ---------------------------------------------
# Split Dataset
# ---------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0
)

# ---------------------------------------------
# Feature Scaling
# ---------------------------------------------
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ---------------------------------------------
# Train Logistic Regression Model
# ---------------------------------------------
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train, y_train)

# ---------------------------------------------
# Prediction
# ---------------------------------------------
y_pred = classifier.predict(X_test)

print("Predicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test)

# ---------------------------------------------
# Confusion Matrix
# ---------------------------------------------
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ---------------------------------------------
# Accuracy Score
# ---------------------------------------------
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", accuracy)

# ---------------------------------------------
# Classification Report
# ---------------------------------------------
from sklearn.metrics import classification_report

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ---------------------------------------------
# Predict New Data
# Example:
# Age = 30
# Estimated Salary = 87000
# ---------------------------------------------
new_prediction = classifier.predict(
    sc.transform([[30, 87000]])
)

print("\nPrediction for [30, 87000]:", new_prediction)


#------------------------
# Visulaise training set
#-------------------------

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1,
              stop=X_set[:, 0].max() + 1,
              step=0.01),
    np.arange(start=X_set[:, 1].min() - 1,
              stop=X_set[:, 1].max() + 1,
              step=0.01)
)

plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(("red", "green"))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(("red", "green"))(i),
        label=j
    )

plt.title("Logistic Regression (Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


#------------------------
# Visulaize test set
#-------------------------

from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(
    np.arange(
        start=X_set[:, 0].min() - 1,
        stop=X_set[:, 0].max() + 1,
        step=0.01
    ),
    np.arange(
        start=X_set[:, 1].min() - 1,
        stop=X_set[:, 1].max() + 1,
        step=0.01
    )
)

plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(("red", "green"))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(("red", "green"))(i),
        label=j
    )

plt.title("Logistic Regression (Test Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


#----------------------------
# Find out value Bias and Variance
#-----------------------------

bias = classifier.score(X_train, y_train)
print(bias)

variance= classifier.score(X_test, y_test)
print(variance)

#------------------------
# Logistic Regression 

# 1) WITHOUT penalty
#--------------------------
## this is simple logisti reression model.

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

# Different Test Sizes
test_sizes = [0.20]

print("-"*70)
print("Test Size Analysis")
print("-"*70)

for size in test_sizes:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=size,
        random_state=0
    )

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    
bias = classifier.score(X_train, y_train)
print(bias)

variance= classifier.score(X_test, y_test)
print(variance)

print(f"\nTest Size : {size}")
print("Training Accuracy :", round(train_acc,4))
print("Testing Accuracy  :", round(test_acc,4))
print("Estimated Bias    :", round(bias,4))
print("Estimated Variance:", round(variance,4))
    
    
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

# Different Test Sizes
test_sizes = [0.25]

print("-"*70)
print("Test Size Analysis")
print("-"*70)

for size in test_sizes:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=size,
        random_state=0
    )

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    
bias = classifier.score(X_train, y_train)
print(bias)

variance= classifier.score(X_test, y_test)
print(variance)

print(f"\nTest Size : {size}")
print("Training Accuracy :", round(train_acc,4))
print("Testing Accuracy  :", round(test_acc,4))
print("Estimated Bias    :", round(bias,4))
print("Estimated Variance:", round(variance,4))
    
    
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

# Different Test Sizes
test_sizes = [0.71]

print("-"*70)
print("Test Size Analysis")
print("-"*70)

for size in test_sizes:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=size,
        random_state=0
    )

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    
bias = classifier.score(X_train, y_train)
print(bias)

variance= classifier.score(X_test, y_test)
print(variance)

print(f"\nTest Size : {size}")
print("Training Accuracy :", round(train_acc,4))
print("Testing Accuracy  :", round(test_acc,4))
print("Estimated Bias    :", round(bias,4))
print("Estimated Variance:", round(variance,4))
    
# Analysis of differant test size with random state =0

# test size = 0.40  => low bias and low variance = underfitting
# test size = 0.71  => high bias and loow variance = overfitting
# test size = 0.20  => high bias and high variance = besst fit

#---------------------------
# Changing Random Sate

# Now keep the test size fixed and change random_state

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, -1].values

# we use random_state = [10, 20, 45, 100, 0]
random_state = [100]

print("_"*50)
print("Random state Analysis")
print("_"*50)

for rs in random_state :
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=rs)
 
    sc = StandardScaler()

X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)    

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

#bias = classifier.score(X_train, y_train)
#print(bias)

#variance= classifier.score(X_test, y_test)
#print(variance)


bias = 1-train_acc
variance = train_acc-test_acc

print(f"\nRandom State : {rs}")
print("Training Accuracy :", round(train_acc,4))
print("Testing Accuracy  :", round(test_acc,4))
print("Estimated Bias    :", round(bias,4))
print("Estimated Variance:", round(variance,4))

# 2) METHOD

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, -1].values

# we use random_state = [20, 45, 100, 0]
random_state = [20]

print("_"*50)
print("Random state Analysis")
print("_"*50)

for rs in random_state :
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=rs)
 
    sc = StandardScaler()

X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)    

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

bias = classifier.score(X_train, y_train)
print(bias)

variance= classifier.score(X_test, y_test)
print(variance)


#bias = 1-train_acc
#variance = train_acc-test_acc

print(f"\nRandom State : {rs}")
print("Training Accuracy :", round(train_acc,4))
print("Testing Accuracy  :", round(test_acc,4))
print("Estimated Bias    :", round(bias,4))
print("Estimated Variance:", round(variance,4))
 
 # 3) METHOD

print("Dataset Loaded")

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

 # we use random_state = [20, 45, 100, 0]
random_state = [45]

print("_"*50)
print("Random state Analysis")
print("_"*50)

for rs in random_state :
     X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=rs)
  
     sc = StandardScaler()

X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)    

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

bias = classifier.score(X_train, y_train)
print(bias)

variance= classifier.score(X_test, y_test)
print(variance)


 #bias = 1-train_acc
 #variance = train_acc-test_acc

print(f"\nRandom State : {rs}")
print("Training Accuracy :", round(train_acc,4))
print("Testing Accuracy  :", round(test_acc,4))
print("Estimated Bias    :", round(bias,4))
print("Estimated Variance:", round(variance,4))
  

# 4) METHOD

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, -1].values

# we use random_state = [20, 45, 100, 0]
random_state = [0]

print("_"*50)
print("Random state Analysis")
print("_"*50)

for rs in random_state :
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=rs)
 
    sc = StandardScaler()

X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)    

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

bias = classifier.score(X_train, y_train)
print(bias)

variance= classifier.score(X_test, y_test)
print(variance)


#bias = 1-train_acc
#variance = train_acc-test_acc

print(f"\nRandom State : {rs}")
print("Training Accuracy :", round(train_acc,4))
print("Testing Accuracy  :", round(test_acc,4))
print("Estimated Bias    :", round(bias,4))
print("Estimated Variance:", round(variance,4))
 
# STEP 3
# Logistic Regression Without Penalty

# No Penalty
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=1000,
    random_state=0
)

model.fit(X_train, y_train)

# L2 Penalty 

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    C= 1.0,
    max_iter = 1000,
    random_state=0)

model.fit(X_train, y_train)

# METHDO 2 (liblinear)

model = LogisticRegression(
    penalty = "l2",
    solver = 'liblinear',
    C=1.0,
    max_iter=1000,
    random_state=0)

model.fit(X_train, y_train)

# METHOD 3 (newton-cg)
model = LogisticRegression(
    penalty="l2",
    solver="newton-cg",
    C=1.0,
    max_iter=1000,
    random_state=0
)

model.fit(X_train, y_train)

# METHOD 4 (newton-cholesky)

model = LogisticRegression(
    penalty="l2",
    solver="newton-cholesky",
    C=1.0,
    max_iter=1000,
    random_state=0
)

model.fit(X_train, y_train)

# METHOD 5 (sag)
model = LogisticRegression(
    penalty="l2",
    solver="sag",
    C=1.0,
    max_iter=3000,
    random_state=0
)

model.fit(X_train, y_train)

# METHOD 5(saga)
model = LogisticRegression(
    penalty="l1",
    solver="saga",
    C=1.0,
    max_iter=3000,
    random_state=0
)

model.fit(X_train, y_train)

# ElasticNet
# only saga support ElasticNet.

model = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.5,
    C=1.0,
    max_iter=5000,
    random_state=0
)

model.fit(X_train, y_train)

# Differant values of C
# Strong Regularisation

model = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    C=0.01,
    max_iter=1000
)

# Medium Regularization

model = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    C=1
)

# Weak Regularization

model = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    C=100
)

#===================================================







