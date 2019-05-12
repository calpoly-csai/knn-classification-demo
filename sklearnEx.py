import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Import data
# CSV with sepal length, sepal width, petal length, petal width, and name
data = pd.read_csv("iris.csv")

# Set test data
# Should be classified as an Iris-virginica
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)


# Running KNN with scikit-learn
print("Using scikit-learn...")
neighbor = KNeighborsClassifier(n_neighbors=3)
neighbor.fit(data.iloc[:,0:4], data['Name'])

# Print the predicted class
print(neighbor.predict(test))

# Print the k nearest neighbors
print(neighbor.kneighbors(test)[1])
