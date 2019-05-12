import pandas as pd
from model import knnModel


#### SETUP
# Import Data
# CSV with sepal length, sepal width, petal length, petal width, and name
data = pd.read_csv("iris.csv")

# Set testSet
# This test data should eb classified as Iris-virginica
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)

#### END SETUP


#### Running our model
# Set # of neighbors = 3
k = 3

# Run our KNN model
print("Using our KNN model...")
result, neighbor = knnModel(data, test, k)

# Print the predicted classification
print(result)

# Prin the k nearest neighbors
print(neighbor)
