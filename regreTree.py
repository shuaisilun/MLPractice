from numpy import  *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

class treeNode():
    def __init__(self, feat, val , right, left):
        featureToSpliOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBrach = left