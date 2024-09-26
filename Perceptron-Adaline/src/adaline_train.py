import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from adaline import AdalineGD

# Preprocessing and splitting the data
def preprocess_and_save_titanic_data(file_path='./data/titanic_data.csv'):
    data = pd.read_csv(file_path)

    # Dropping irrelevant features
    data_cleaned = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Converting categorical variables to numeric
    data_cleaned['Sex'] = data_cleaned['Sex'].map({'male': 0, 'female': 1})
    data_cleaned['Embarked'] = data_cleaned['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Filling missing values
    data_cleaned.loc[:, 'Age'] = data_cleaned['Age'].fillna(data_cleaned['Age'].median())
    data_cleaned.loc[:, 'Embarked'] = data_cleaned['Embarked'].fillna(data_cleaned['Embarked'].median())

    # Splitting features and target
    X = data_cleaned.drop('Survived', axis=1)
    y = data_cleaned['Survived']

    # Splitting the data into 70% training and 30% testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test  # Return as DataFrames, not NumPy arrays

# Load and preprocess the data
X_train, X_test, y_train, y_test = preprocess_and_save_titanic_data()

# Initialize and train the Adaline model
adaline = AdalineGD(eta=0.0001, n_iter=75)
adaline.fit(X_train.values, y_train.values)  # Convert to NumPy arrays only for fitting

# Predicting the labels for the training and test sets
y_train_pred = adaline.predict(X_train.values)
y_test_pred = adaline.predict(X_test.values)

# Calculate accuracy for training and test sets
train_accuracy = (y_train_pred == y_train.values).mean()
test_accuracy = (y_test_pred == y_test.values).mean()

print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Generate random predictions
np.random.seed(42)  # For reproducibility
random_predictions = np.random.randint(0, 2, size=len(y_test))

# Calculate the accuracy of the random baseline model
random_accuracy = accuracy_score(y_test, random_predictions)
print(f'Random Baseline Accuracy: {random_accuracy * 100:.2f}%')