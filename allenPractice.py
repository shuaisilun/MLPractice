from numpy import *
import matplotlib.pyplot as plt

def loadfiles(fname):
    datamat= [] ; lablemat = []
    numOfFeature = len(open(fname).readline().split('\t')) - 1
    fr = open(fname)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numOfFeature):
            lineArr.append(float(curLine[i]))
        datamat.append(lineArr)
        lablemat.append(float(curLine[-1]))
    return datamat, lablemat


def standardRegres(d, l):
    dmat = mat(d) ; lablemat= mat(l).T
    m, n = dmat.shape
    w = ones(shape = (m,1))
    xTx = dmat.T * dmat
    if linalg.det(xTx) == 0:
        print("the xTx matrix is singular, can not find inverse!")
        return
    w = xTx.I * (dmat.T * lablemat)
    return w


def rigRegerss(d, l, lamda = 0.2):
    dmat = mat(d) ; lablemat= mat(l).T
    m, n = dmat.shape
    ymean = mean(lablemat, 0)
    xmean = mean(dmat, 0)
    xVar = var(dmat, 0)
    lablemat = lablemat - ymean
    lablemat = (lablemat - xmean) /xVar
    w = ones(shape = (m,1))
    xTx = dmat.T * dmat + lamda* eye(n)
    if linalg.det(xTx) == 0:
        print("the xTx matrix is singular, can not find inverse!")
        return
    w = xTx.I * (dmat.T * lablemat)
    return w


def lwlr(testpoint, d, l, k = 1.0):
    dmat = mat(d) ; lablemat= mat(l).T
    m, n = dmat.shape
    wz = mat(eye(m))
    for i in range(m):
        diff = dmat[i, :] - testpoint
        wz[i,i] = exp((diff * diff.T) / (-2*k**2))
    ws = ones(shape = (m,1))
    xTx = dmat.T * (wz * dmat)
    if linalg.det(xTx) == 0:
        print("the xTx matrix is singular, can not find inverse!")
        return
    ws = xTx.I * (dmat.T * (wz * lablemat))
    return testpoint * ws


#####################################test#code###################################################
dmat, lmat = loadfiles('ex0.txt')

#w = standardRegres(dmat, lmat)
#w = rigRegerss(dmat, lmat, 0.3)
#predictResults = dmat * w
#predictResults = predictResults.flatten()
#lmat = mat(lmat)

predictResults = []
dmat= mat(dmat)
for i in range(len(dmat)):
    predictResults.append(lwlr(dmat[i, :], dmat, lmat)[0, 0])

similarity = corrcoef(predictResults, lmat)
print("Similarity is %f" %similarity[0][1])


