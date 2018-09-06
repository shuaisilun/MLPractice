import math
from numpy import *

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels


def calcShannonEnt(dataset):
    numEntries = len(dataset)
    lableConts = {}
    for iter in dataset:
        currentlable = iter[-1]     #assuming last one is the lable
        if currentlable not in lableConts.keys():
            lableConts[currentlable] = 0
        lableConts[currentlable] += 1
    shannonEnt = 0.0
    entlist = {}
    for key in lableConts:
        prob = float(lableConts[key])/numEntries
        entlist[key] = prob * math.log(prob, 2)
    entvals = list(entlist.values())
    shannonEnt = array(entvals).sum()
    return -shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

