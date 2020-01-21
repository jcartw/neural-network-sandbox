import math
import numpy as np
# import matplotlib.pyplot as plt

# Based on: "Can Machine Learn the Concept of Sine" by Ying Xie
# Medium Link: https://towardsdatascience.com/can-machine-learn-the-concept-of-sine-4047dced3f11
# GitHub: https://github.com/looselyconnected/ml-examples

# Goal: create ANN to learn generalized model 'y = sin(A*x)'

INPUT_COUNT = 2

a = 0.06
train_count = 6 
test_count = 4

def getSineData(a, count):
    return np.array([math.sin(a * i) for i in range(0, count)])

# def getSineData(a, count):
#    return np.array([i*1 for i in range(0, count)])

def getXY(data, total):
    x = data[0: total - INPUT_COUNT].reshape((total - INPUT_COUNT, 1))
    for i in range(1, INPUT_COUNT):
        x = np.concatenate((x, data[i: total - INPUT_COUNT + i].reshape((total - INPUT_COUNT, 1))), axis=1)
    
    y = data[INPUT_COUNT : total]
    return x,y

def getSeries(a, trainCount, testCount=0):
    data = getSineData(a, trainCount+testCount)
    x,y = getXY(data, trainCount)
    if testCount > INPUT_COUNT:
        evalX, evalY = getXY(data[trainCount:], testCount)
    else:
        evalX, evalY = None, None
    return x, y, evalX, evalY

def getCNNSeries(a, trainCount, testCount):
    x, y, evalX, evalY = getSeries(a, trainCount, testCount)
    if testCount <= INPUT_COUNT:
        return x.reshape(x.shape[0], x.shape[1], 1), y, None, None
    return x.reshape(x.shape[0], x.shape[1], 1), y, evalX.reshape(evalX.shape[0], evalX.shape[1], 1), evalY


# -------------------------------------------------------------------
# data = getSineData(a=a, count=train_count + test_count)
# -------------------------------------------------------------------
# an array of sine data with angular frequency 'a' and length = train_count + test_count

# -------------------------------------------------------------------
# data = getSineData(a=a, count=train_count + test_count)
# x, y = getXY(data=data, total=train_count)
# evalX, evalY = getXY(data=data[train_count:], total=test_count)
# -------------------------------------------------------------------
# x is a 2D array of (train_count - INPUT_COUNT) X (INPUT_COUNT)
#   where each row is an array of shifted time series values
# y is a 1D array of length (train_count - INPUT_COUNT) where each value is the succeeding 
#   value of row 'k' from the 2D array X
# evalX is a 2D array of (test_count - INPUT_COUNT) X  (INPUT_COUNT)
#   where each row is an array of shifted time series values
# evalY is a 1D array of length (train_count - INPUT_COUNT) where each value is the succeeding 
#   value of row 'k' from the 2D array X

# TRAIN SET SIZE:  (train_count - INPUT_COUNT)
# TEST SET SIZE:   (test_count - INPUT_COUNT)
# TOTAL SAMPLES:   train_count + test_count - 2*INPUT_COUNT

# EX: train_count = 6, test_count = 4, INPUT_COUNT=2
# source_data    = [0 1 2 3 4 5 6 7 8 9]
# x_train (x)    = [[0 1], [1 2], [2 3], [3 4]]
# y_train (y)    = [   2,     3,     4,     5 ]
# x_test (evalX) = [[6 7], [7 8]]
# y_test (evalY) = [   8,     9 ]


# -------------------------------------------------------------------------------
# x, y, evalX, evalY = getSeries(a=a, trainCount=train_count, testCount=test_count)
# -------------------------------------------------------------------------------
# see above for x, y, evalX, and evalY


# ------------------------------------------------------------------------------------
x, y, evalX, evalY = getCNNSeries(a=a, trainCount=train_count, testCount=test_count)
# ------------------------------------------------------------------------------------
# takes 'x' and 'evalX' from 'getSeries' and stuffs each value into 1x1 array
# for example: before x.shape=(4,2), after x.shape=(4,2,1) 

print("")
print("x")
print(x.shape)
print(x)
print("")
print("y")
print(y.shape)
print(y)
print("")
print("evalX")
print(evalX.shape)
print(evalX)
print("")
print("evalY")
print(evalY.shape)
print(evalY)
print("")