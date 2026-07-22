# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv(
    r"E:\ML\Machine learning\ML_Classification\Logistic_Regression_Project\data\logistic_classification.csv"
)

# Display first 5 rows
print(dataset.head())

# Independent and Dependent variables
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=0
)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# ==============================================================
# Training the Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)   # guassianNB() naive bayes gives High score =0.92
                                    # it is applicable for both with or without features scaling

# Training the Bernoulli Naive Bayes model
from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()
classifier.fit(X_train, y_train)   # BernoulliNB() give score = 0.82


# Training the Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X_train, y_train)  # MultinomialNB() give score very low = 0.56
                                  # It does not allow standered scaling .

#=======================================================================


# Prediction
y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy
from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)

# Training Accuracy (Bias)
bias = classifier.score(X_train, y_train)
print("Training Accuracy (Bias):", bias)

# Testing Accuracy (Variance)
variance = classifier.score(X_test, y_test)
print("Testing Accuracy (Variance):", variance)