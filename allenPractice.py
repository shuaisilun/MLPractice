from numpy import *
from math import *
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


def standardRegressPredict(testsample, w):
    return float(testsample*w)


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

class weakClassifier:
    def __init__(self, trainfunc,  predictfunc):
        self.predictfunc = predictfunc
        self.trainfunc = trainfunc
        self.w = 0
    def train(self, dataset, lables):
        self.w = self.trainfunc(dataset,lables)
    def predict(self, testsample):
        return self.predictfunc(testsample, self.w)

def adaboost(dataset, lables, K = 10): ### This is from pinard's blog
    dmat = mat(dataset) ; lmat = mat(lables)
    larr = array(lables)
    m,n = dmat.shape
    w = ones(shape = (m, 1))
    w = w/m
    alphas = ones(shape = (K, 1))
    weakclist = []
    for i in range(K):
        dmat = multiply(w,dmat)
        wc = weakClassifier(standardRegres, standardRegressPredict)
        predictresults = []
        wc.train(dataset, lables)
        for j in range(m):
            predictresults.append(wc.predict(dataset[j]))
        diff = array(predictresults) - larr
        errorRate = len(diff[diff >= 0.01]) / m
        if errorRate <= 0.3:
            return weakclist, alphas
        alphas[i] = math.log((1-errorRate)/errorRate, 10) /2
        z =0
        for j in range(m):
            z += w[j]*exp(-alphas[i]*diff[j])
        for j in range(m):
            w[j] = w[j] / z * exp(-alphas[i] * (lables[j]*predictresults[j]))
        weakclist.append(wc)
    return weakclist, alphas


def adaboostTest(weakclist, alphas, testsample):
    t = len(weakclist)
    result = 0
    for i in range(t):
        result += alphas[i]*weakclist[i].predict()
    return sign(result)


def svd(dmat):
    dmat = mat(dmat)
    m, n = shape(dmat)
    mtm = dmat.T * dmat
    eigValmtm, eigVectmtm = linalg.eig(mtm)
    v = eigVectmtm
    mmt = dmat * dmat.T
    eigValmmt, eigVectmmt = linalg.eig(mmt)
    u = eigVectmmt
    num = min(eigVectmmt.shape[1], eigVectmtm.shape[1])
    s = zeros(shape = (m, n))
    for i in range(num):
        s[i, i] = (dmat * v[:, i] / u[:, i])[0, 0]
    return u, s, v


#####################################test#code###################################################
def testLinearRegression():
    dmat, lmat = loadfiles('ex0.txt')
    w = standardRegres(dmat, lmat)
    #w = rigRegerss(dmat, lmat, 0.3)
    predictResults = dmat * w
    predictResults = predictResults.flatten()
    lmat = mat(lmat)
    similarity = corrcoef(predictResults, lmat)
    print("Similarity is %f" %similarity[0][1])

def testlwlr():
    dmat, lmat = loadfiles('ex0.txt')
    predictResults = []
    dmat= mat(dmat)
    for i in range(len(dmat)):
        predictResults.append(lwlr(dmat[i, :], dmat, lmat)[0, 0])
    lmat = mat(lmat)
    similarity = corrcoef(predictResults, lmat)
    print("Similarity is %f" %similarity[0][1])


def testAdaboost():
    d, l = loadfiles('ex0.txt')
    adaboost(d,l)


#testAdaboost()
m = mat([[1,1],[7,7]])
u1, sig1, v1 = svd(m)
print(u1)
print(sig1)
print(v1)
