# EXPERIMENT 5: K-NEAREST NEIGHBOURS

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Use first two features for visualization
X_visual = X[:, :2]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_visual,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# TEST DIFFERENT K VALUES
# -----------------------------

k_values = range(1, 16)

accuracies = []

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy)

# Find best K
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)

print("Best K:", best_k)
print("Best Accuracy:", best_accuracy)


# -----------------------------
# GRAPH 1
# ACCURACY VS K
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    k_values,
    accuracies,
    marker='o'
)

plt.title("K-NN Accuracy for Different K Values")
plt.xlabel("Value of K")
plt.ylabel("Accuracy")

plt.xticks(list(k_values))
plt.grid(True)

plt.show()


# -----------------------------
# TRAIN FINAL KNN MODEL
# -----------------------------

knn = KNeighborsClassifier(
    n_neighbors=best_k
)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("\nFinal K-NN Accuracy:")
print(accuracy_score(y_test, y_pred))


# -----------------------------
# GRAPH 2
# CLASSIFICATION SCATTER PLOT
# -----------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_pred,
    s=80
)

plt.title("K-NN Classification")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.grid(True)

plt.show()
