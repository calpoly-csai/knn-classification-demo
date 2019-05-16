import pandas as pd
import numpy as np
import math
import operator

#### BUILDING YOUR OWN MODEL

# Calculates euclidean distance between points in n dimensions
def euclideanDistance(p1, p2, n):
   distance = 0

   #### Part 1 - return euclidean distance between p1 and p2

   #### End of Part 1

# Defining our own KNN model
def knnModel(trainingSet, testingSet, k):
   # Initialize dictionary
   distances = {}

   # Note: "shape" is an attribute for numpy arrays, returning the dimensions of the array in the form n rows x m columns
   # in this case, .shape[1] returns the # of columns
   length = testingSet.shape[1]

   #### Part 2 - Find euclidean distance between each row of training data and test data

      #### End of Part 2
      

   #### Part 3 - Sort distances

   #### End of Part 3


   # Getting top k neighbors
   neighbors = []


   #### Part 4 - Get top k neighbors

   #### End of Part 4

   # Getting most frequent class in neighbors
   classVotes = {}
   # Iterate through top k neighbors
   for x in range(len(neighbors)):
      # get class of each neighbor
      response = trainingSet.iloc[neighbors[x]][-1]
  
      #### Part 5 - getting votes for each class

      #### End of Part 5


   # set sortedVotes to the sorted classes with largest value first
   sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

   # return the the most voted class (the classification prediction!)
   return(sortedVotes[0][0], neighbors)
