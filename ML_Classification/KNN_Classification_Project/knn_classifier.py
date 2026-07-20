# =============================================================================
# K-Nearest Neighbors (K-NN) Classification
# =============================================================================

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Step 1 : Import Dataset
# =============================================================================

dataset = pd.read_csv(r"data/Social_Network_Ads.csv")

# Independent Variables
X = dataset.iloc[:, [2, 3]].values

# Dependent Variable
y = dataset.iloc[:, -1].values

# =============================================================================
# Step 2 : Split Dataset into Training and Test Set
# =============================================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)

# =============================================================================
# Step 3 : Feature Scaling
# =============================================================================

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# Step 4 : Train the K-NN Model
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',
    p=2
)

classifier.fit(X_train, y_train)

# =============================================================================
# Step 5 : Predict Test Set Results
# =============================================================================

y_pred = classifier.predict(X_test)

# =============================================================================
# Step 6 : Confusion Matrix
# =============================================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("="*50)
print("Confusion Matrix")
print("="*50)
print(cm)

# =============================================================================
# Step 7 : Accuracy
# =============================================================================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy, 4))

# =============================================================================
# Step 8 : Classification Report
# =============================================================================

from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)

print("\nClassification Report")
print(report)

# =============================================================================
# Step 9 : Bias and Variance
# =============================================================================

bias = classifier.score(X_train, y_train)

variance = classifier.score(X_test, y_test)

print("\nTraining Accuracy (Bias) :", round(bias, 4))
print("Testing Accuracy (Variance) :", round(variance, 4))


# =============================================================================
# Step 10 : ROC Curve
# =============================================================================

from sklearn.metrics import roc_auc_score, roc_curve

y_prob = classifier.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)

print("\nAUC Score :", round(auc, 4))

fpr, tpr, threshold = roc_curve(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# Step 11 : Visualising Training Set
# =============================================================================

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(
    np.arange(X_set[:,0].min()-1, X_set[:,0].max()+1, 0.01),
    np.arange(X_set[:,1].min()-1, X_set[:,1].max()+1, 0.01)
)

plt.figure(figsize=(8,6))

plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.4,
    cmap=ListedColormap(("red","green"))
)

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set==j,0],
        X_set[y_set==j,1],
        label=j
    )

plt.title("K-NN Classification (Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

# =============================================================================
# Step 12 : Visualising Test Set
# =============================================================================

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(
    np.arange(X_set[:,0].min()-1, X_set[:,0].max()+1, 0.01),
    np.arange(X_set[:,1].min()-1, X_set[:,1].max()+1, 0.01)
)

plt.figure(figsize=(8,6))

plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.4,
    cmap=ListedColormap(("red","green"))
)

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set==j,0],
        X_set[y_set==j,1],
        label=j
    )

plt.title("K-NN Classification (Test Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()