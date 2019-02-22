#version 2.0 started implementing backward elimination and custom algorithm
import numpy as np
import os
import copy
import random
import time
import math


#ALWAYS GET 3 FEATURES EVEN IF TWO ARE STRONGEST

def leaveOneOutCrossValidation(data, classCol, current_set, feature_to_add):
    current_set.append(feature_to_add)
    curDistance = 0.0
    minDistance = 100000.0 #arbitrary value
    sum = 0.0
    hitsArray = []
    for i in range(0, len(classCol)): #loop to pick each test point to be left out
        hits = 0
        minDistance = 100000000  #arbitrary reset for min distance
        curDistance = 0.00
        for j in range(0, len(classCol)): #this is the actual test for each point
            if j != i: #this means we can check perform the distance calculation
                sum = 0.0 #reset sum for each testing point's distance
                for k in range(0, len(current_set)):
                    difference = matrix[j][current_set[k]] - matrix[i][current_set[k]]
                    square = difference**2
                    sum += square

                curDistance = math.sqrt(sum)
                if curDistance < minDistance:
                    minDistance = curDistance
                    nearestNeighbor = j
                if classCol[i] == classCol[nearestNeighbor]:
                    hits = hits + 1 #successful guesses
        hitsArray.append(hits)

    current_set.remove(feature_to_add)
    return np.sum(hitsArray)/len(hitsArray) / len(classCol)

def backwardsLeaveOneOutCrossValidation(data, classCol, current_set, feature_to_remove):
    if current_set: #if there actually is a feature to remove
        current_set.remove(feature_to_remove)

    curDistance = 0.0
    minDistance = 100000.0 #arbitrary value
    sum = 0.0
    hitsArray = []
    for i in range(0, len(classCol)): #loop to pick each test point to be left out
        hits = 0
        minDistance = 100000000  #arbitrary reset for min distance
        curDistance = 0.00
        for j in range(0, len(classCol)): #this is the actual test for each point
            if j != i: #this means we can check perform the distance calculation
                sum = 0.0 #reset sum for each testing point's distance
                for k in range(0, len(current_set)):
                    difference = matrix[j][current_set[k]] - matrix[i][current_set[k]]
                    square = difference**2
                    sum += square

                curDistance = math.sqrt(sum)
                if curDistance < minDistance:
                    minDistance = curDistance
                    nearestNeighbor = j
                if classCol[i] == classCol[nearestNeighbor]:
                    hits = hits + 1 #successful guesses
        hitsArray.append(hits)

    current_set.append(feature_to_remove)
    return np.sum(hitsArray)/len(hitsArray) / len(classCol)

def backwardElimination(matrix, classCol):
    current_set_of_features = []
    best_set_of_features = []
    bestAccuracy = 0.0
    totalAccuracy = 0.0
    featureCount = matrix.shape[1]

    for i in range (0, matrix.shape[1]):
        current_set_of_features.append(i)
    # FEATURE SEARCH TREE FOR BACKWARD ELIMINATION
    print("Beginning search.")
    for i in range(0, matrix.shape[1]):  # matrix.shape returns tuple (# rows, #columns)
        featureToRemove = [];
        bestAccuracy = 0.0;
        for j in range(0, matrix.shape[1]):
            if j in current_set_of_features:
                accuracy = backwardsLeaveOneOutCrossValidation(matrix, classCol, current_set_of_features, j)
                print("Removing feature " + str(j+1) + " gives an accuracy of " + str(accuracy))
                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    featureToRemove = j
                    best_set_of_features = copy.deepcopy(current_set_of_features)

                if featureCount == 3:
                    featureToRemove = j

                    best_set_of_features.remove(featureToRemove)
                    bestAccuracy = accuracy
                    featureCount -= 1
        print("\nRemoving feature " + str(featureToRemove+1) + " is best, accuracy is " + str(bestAccuracy) + "\n")
        if featureCount > 3:
            current_set_of_features.remove(featureToRemove)
            featureCount -=1
        if featureCount == 3:
            break

    for i in range(0, len(current_set_of_features)):
        current_set_of_features[i] += 1
    print("Finished search! The best feature subset is " + str(current_set_of_features) + ", which has an accuracy of " + str(bestAccuracy))

def rangelLeaveOneOutCrossValidation(data, classCol, current_set, feature_to_add): #drop features after certain number of misses?
    current_set.append(feature_to_add)
    curDistance = 0.0
    minDistance = 100000.0 #arbitrary value
    sum = 0.0
    hitsArray = []
    missesArray = []
    zeroFlag = False
    for i in range(0, len(classCol)): #loop to pick each test point to be left out
        hits = 0
        misses = 0
        minDistance = 100000000  #arbitrary reset for min distance
        curDistance = 0.00
        for j in range(0, len(classCol)): #this is the actual test for each point
            if j != i: #this means we can check perform the distance calculation
                sum = 0.0 #reset sum for each testing point's distance
                for k in range(0, len(current_set)):
                    difference = matrix[j][current_set[k]] - matrix[i][current_set[k]]
                    square = difference**2
                    sum += square

                curDistance = math.sqrt(sum)
                if curDistance < minDistance:
                    minDistance = curDistance
                    nearestNeighbor = j
                if classCol[i] == classCol[nearestNeighbor]:
                    hits = hits + 1 #successful guesses
                else:
                    misses = misses + 1

        hitsArray.append(hits)
        missesArray.append(misses)
        missCount = 0
        #loop through and check if there are more misses than hits
        for i in range(0, len(hitsArray)):
            if missesArray[i] > hitsArray [i]: #more misses than hits for the given test point
                missCount += 1
            if missCount > 50:
                zeroFlag = True
                break

    current_set.remove(feature_to_add)
    if zeroFlag:
        return 0
    else:
        return np.sum(hitsArray)/len(hitsArray) / len(classCol)


def rangelSelection(matrix, classCol):
    current_set_of_features = []
    best_set_of_features = []
    bestAccuracy = 0.0;
    totalAccuracy = 0.0;
    ZeroFlag = False
    featureCount = 0
    # FEATURE SEARCH TREE FOR RANGEL ELECTION
    print("Beginning search.")
    for i in range(0, matrix.shape[1]):  # matrix.shape returns tuple (# rows, #columns)
        featureToAdd = [];
        bestAccuracy = 0.0;
        for j in range(0, matrix.shape[1]):
            ZeroFlag = False
            if not (j in best_set_of_features):
                accuracy = rangelLeaveOneOutCrossValidation(matrix, classCol, best_set_of_features, j)
                tempset = []
                for i in range(0, len(best_set_of_features)):
                    tempset.append(best_set_of_features[i] + 1)
                print("Using feature(s) " + str(tempset) + " and " + str(j + 1) + " accuracy is " + str(accuracy))
                if accuracy == 0:
                    ZeroFlag == True
                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    featureToAdd = j
                    best_set_of_features = copy.deepcopy(current_set_of_features)
                    best_set_of_features.append(featureToAdd)
                    totalAccuracy = accuracy
                if featureCount == 2:  # add one feature even if it isn't strong then stop
                    featureToAdd = j
                    totalAccuracy = accuracy
                    featureCount += 1

        tempset2 = []
        for i in range(0, len(best_set_of_features)):
            tempset2.append(best_set_of_features[i] + 1)
        print("\nFeature set " + str(tempset2) + " is best, accuracy is " + str(totalAccuracy) + "\n")

        if featureCount < 3 and not ZeroFlag:
            current_set_of_features.append(featureToAdd)
            featureCount += 1
        if featureCount == 3:
            break
    for i in range(0, len(best_set_of_features)):
        best_set_of_features[i] += 1
    print("Finished search! The best feature subset is " + str(best_set_of_features) + ", which has an accuracy of " + str(totalAccuracy))


def forwardSelection(matrix, classCol):
    current_set_of_features = []
    best_set_of_features = []
    bestAccuracy = 0.0
    totalAccuracy = 0.0
    featureCount = 0
    # FEATURE SEARCH TREE FOR FORWARD SELECTION
    print("Beginning search.")
    for i in range(0, matrix.shape[1]):  # matrix.shape returns tuple (# rows, #columns)
        # print("On level " + str(i + 1) + " of tree")
        featureToAdd = [];
        bestAccuracy = 0.0;
        for j in range(0, matrix.shape[1]):
            if not (j in current_set_of_features):
                accuracy = leaveOneOutCrossValidation(matrix, classCol, current_set_of_features, j)
                tempset = []
                for i in range(0, len(current_set_of_features)):
                    tempset.append(current_set_of_features[i] + 1)
                print("Using feature(s) " + str(tempset) + " and " + str(j + 1) + " accuracy is " + str(accuracy))
                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    featureToAdd = j
                    best_set_of_features = copy.deepcopy(current_set_of_features)
                    best_set_of_features.append(featureToAdd)
                    totalAccuracy = accuracy
                if featureCount == 2: #add one feature even if it isn't strong then stop
                    featureToAdd = j
                    best_set_of_features.append(featureToAdd)
                    totalAccuracy = accuracy
                    featureCount += 1

        tempset2 = []
        for i in range(0, len(best_set_of_features)):
            tempset2.append(best_set_of_features[i] + 1)
        print("\nFeature set " + str(tempset2) + " is best, accuracy is " + str(totalAccuracy) + "\n")
        if featureCount < 3:
            current_set_of_features.append(featureToAdd)
            featureCount += 1
        if featureCount == 3:
            break

    for i in range(0, len(best_set_of_features)):
        best_set_of_features[i] += 1
    print("Finished search! The best feature subset is " + str(best_set_of_features) + ", which has an accuracy of " + str(totalAccuracy))


print("Welcome to Richard Rangel's Feature Selection Algorithm")
print(os.listdir(os.getcwd()))
fileName = input("Type in the name of the file to test: ")
matrix = np.loadtxt(fileName, dtype='float')
#matrix = np.loadtxt("CS170_LARGEtestdata__108.txt", dtype='float')
classCol = matrix[:,0] #store the first column of the matrix
matrix = np.delete(matrix, 0, 1) #delete the first column of the matrix
print("1 - Forward Selection\n2 - Backward Elimination\n3 - Richard's Custom Algorithm\n")
algo = input("Type the number of the algorithm you want to run: ")

if algo == "1": #forward selection
    startTime = time.time()
    forwardSelection(matrix, classCol)
elif algo == "2": #backward elimination
    startTime = time.time()
    backwardElimination(matrix, classCol)
elif algo == "3": #custom algorithm
    startTime = time.time()
    rangelSelection(matrix, classCol)

finishTime = time.time() - startTime
print("Program runtime = " + str(finishTime))