from numpy import *

import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(filename, delim= '\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datamat = [list(map(float, line)) for line in stringArr]
    return mat(datamat)


def pca(dataSet, topNfeat = 9999999):
    meanVals = mean(dataSet, axis =0)
    meanRemoved = dataSet - meanVals
    covMat = cov(meanRemoved, rowvar = 0)
    eigVal , eigVector = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVal)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVector[:, eigValInd]
    lowdatamat = meanRemoved * redEigVects
    reconMat = (lowdatamat * redEigVects.T) + meanVals
    return lowdatamat, reconMat


###################################################################################
def testcode():
    datamat = loadDataSet('testSet.txt')
    lowmat, reconmat = pca(datamat, 1)
    print(shape(lowmat))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datamat[:, 0].flatten().A[0], datamat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconmat[:, 0].flatten().A[0], reconmat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

testcode()
