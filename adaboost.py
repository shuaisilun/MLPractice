from numpy import *

def loadSimpData():
    datamat = matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classlable = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datamat, classlable

def stumpClassify(datamatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(datamatrix)[0],1))
    if threshIneq == 'lt':
        retArray[datamatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[datamatrix[:, dimen] > threshVal] = -1.0
    return retArray


