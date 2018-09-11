from numpy import *


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix:
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights


def stocGrandAscent0(data, classlable):
    datamat = mat(data)
    m,n= shape(datamat)
    alpha = 0.01
    weights = ones((n,1))
    for i in range(m):
        h = sigmoid(sum(datamat[i]*weights))
        error = classlable[i]-h
        weights = weights + alpha*datamat[i].transpose()*error
    return weights


def stocGrandAscent1(data, classlable, numIter = 100):
    datamat = mat(data)
    m,n= shape(datamat)
    weights = ones((n,1))
    for i in range(numIter):
        dataIndex = list(range(0,m))
        for j in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randindex = random.choice(dataIndex)
            h = sigmoid(sum(datamat[randindex]*weights))
            error = classlable[randindex]-h
            weights = weights + alpha*datamat[randindex].transpose()*error
            del(dataIndex[dataIndex.index(randindex)])
    return weights


def classify0(data,weights):
    h = sigmoid(sum(data*weights))
    if h>=0.5:
        return 1
    else:
        return 0

######################################################################
#return accuracy of correct classify

def testWeights(data,lables,weights):
    testlable = []
    datamat= mat(data)
    for i in range(datamat.shape[0]):
        testlable.append(classify0(data[i],weights))
    return 1-bitwise_xor(testlable,lables).sum()/len(data)

m,l = loadDataSet()
weights = gradAscent(m,l)
accuracy = testWeights(m,l,weights)
print("gradAscent (alpha = 0.001,iter=500) accuracy = %f" % accuracy)

weights = stocGrandAscent0(m,l)
accuracy = testWeights(m,l,weights)
print("stocGrandAscent0 (alpha = 0.01,iter=100) accuracy = %f" % accuracy)

weights = stocGrandAscent1(m,l)
accuracy = testWeights(m,l,weights)
print("stocGrandAscent1 (alpha = random,iter=100) accuracy = %f" % accuracy)
