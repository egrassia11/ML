import numpy as np
import pandas as pd
from logistic_regression import LogisticRegressionModel

# Data preparation
data = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
test_ids = test["PassengerId"]

def clean(data):
    data = data.drop(["Ticket", "PassengerId", "Name", "Cabin"], axis=1)

    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols:
        data[col] = data[col].fillna(data[col].median())

    data["Embarked"] = data["Embarked"].fillna("U")
    return data

data = clean(data)
test = clean(test)

# Convert categorical columns into numerical values
columns = ["Sex", "Embarked"]
for column in columns:
    data[column] = pd.Categorical(data[column]).codes
    test[column] = pd.Categorical(test[column]).codes

y = data["Survived"].values
X = data.drop("Survived", axis=1).values

# Split the data into training and validation sets
def train_test_split(X, y, test_size=0.2):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Initialize and train the model
model = LogisticRegressionModel(learning_rate=0.01, iterations=1000)
model.train(X_train, y_train)

# Make predictions on the validation set
predictions = model.predict(X_val)

# Calculate accuracy
accuracy = model.accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
