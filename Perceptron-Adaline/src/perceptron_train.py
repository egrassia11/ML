from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Create a linearly separable dataset
X = np.array([
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
    [6, 7],
    [7, 8],
    [8, 8],
    [9, 10],
    [10, 12]
])

y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

# Initialize and train the perceptron
perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X, y)

# Predict the training set
predictions = perceptron.predict(X)

# Measure and print the accuracy
accuracy = accuracy_score(y, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Evidence that the perceptron found a decision boundary
x_values = [i for i in range(0, 12)]
y_values = -(perceptron.weights[0] * np.array(x_values) + perceptron.bias) / perceptron.weights[1]

plt.figure(figsize=(8, 6))
plt.scatter(X[:5, 0], X[:5, 1], color='red', marker='o', label='Class -1')
plt.scatter(X[5:, 0], X[5:, 1], color='blue', marker='x', label='Class 1')
plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Perceptron Decision Boundary")

# Optionally generate a graph
# plt.savefig('./plots/perceptron_boundary.png')