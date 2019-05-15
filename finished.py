import pandas as pd
import numpy as np
import math
import operator

#### BUILDING YOUR OWN MODEL

# Calculates euclidean distance between points in n dimensions
def euclideanDistance(p1, p2, n):
   distance = 0
   for x in range(n):
      distance += np.square(p1[x] - p2[x])

   return np.sqrt(distance)

# Defining our own KNN model
def knnModel(trainingSet, testingSet, k):
   # Initialize dictionaries
   distances = {}
   sort = {}

   # Note: "shape" is an attribute for numpy arrays, returning the dimensions of the array in the form n rows x m columns
   # in this case, .shape[1] returns the # of columns
   length = testingSet.shape[1]

   # Find euclidean distance between each row of training data and test data
   for x in range(len(trainingSet)):
      # pandas method to grab data by index
      dist = euclideanDistance(testingSet, trainingSet.iloc[x], length)
      
      # Have to use array notation because numpy's sqrt function returns an array!
      distances[x] = dist[0]

   
   # Sorts distances
   # sorted() function takes in an interable object and key (way to sort the iterable object)
   # operator.itemgetter(1) is a way to grab the first item from the iterable object
   sortedDistances = sorted(distances.items(), key=operator.itemgetter(1))

   # Getting top k neighbors
   neighbors = []
   for x in range(k):
      neighbors.append(sortedDistances[x][0])

   # Getting most frequent class in neighbors
   classVotes = {}
   # Iterate through top k neighbors
   for x in range(len(neighbors)):
      # get class of each neighbor
      response = trainingSet.iloc[neighbors[x]][-1]
  
      # if the class is already in the array, increment
      if response in classVotes:
         classVotes[response] += 1
      # else add a new counter to the array
      else:
         classVotes[response] = 1

   # set sortedVotes to the sorted classes with largest value first
   sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

   # return the the most voted class (the classification prediction!)
   return(sortedVotes[0][0], neighbors)
