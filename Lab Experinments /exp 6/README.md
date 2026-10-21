

# EXPERIMENT 6: NAIVE BAYES CLASSIFICATION
# Google Colab

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# STEP 1: Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target
class_names = iris.target_names

print("Dataset loaded successfully!")
print("Total samples:", len(X))
print("Features:", iris.feature_names)
print("Classes:", class_names)

# STEP 2: Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 3: Create and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

print("\nNaive Bayes model trained successfully!")

# STEP 4: Predict the test data
y_pred = model.predict(X_test)

print("\nActual classes:")
print(y_test)

print("\nPredicted classes:")
print(y_pred)

# STEP 5: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nClassification Accuracy: {:.2f}%".format(
    accuracy * 100
))

# STEP 6: Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# STEP 7: Plot confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Naive Bayes - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# STEP 8: Plot accuracy graph
plt.figure(figsize=(6, 4))

plt.bar(
    ["Naive Bayes"],
    [accuracy * 100]
)

plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Naive Bayes Classification Accuracy")

for i, value in enumerate([accuracy * 100]):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()

# STEP 9: Display classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names
))
