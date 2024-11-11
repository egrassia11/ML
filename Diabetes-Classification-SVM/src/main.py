import pandas as pd
from sklearn.model_selection import train_test_split
from support_vector_machine import SupportVectorMachine, cross_validate, custom_grid_search
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = pd.read_csv('./data/diabetes.csv')

# Split dataset into features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for SVM
param_grid = {
    "SVM": {
        "model": SupportVectorMachine,
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    },
}

# Perform grid search
best_model, best_params, best_score = custom_grid_search(X_train, y_train, param_grid, n_folds=5)
print(f"Best Model: {best_model}")
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validated Score: {best_score:.4f}")

# Train and evaluate the best model on the test set
final_model = best_model(**best_params)
final_model.fit(X_train, y_train)
test_accuracy = final_model.score(X_test, y_test)
print(f"Test Set Accuracy with Best Parameters: {test_accuracy:.4f}")

# Evaluate the best model on the training set
train_accuracy = final_model.score(X_train, y_train)
print(f"Training Set Accuracy with Best Parameters: {train_accuracy:.4f}")
