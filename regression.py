from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


def standRegression(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat;
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot find inverser")
        return
    ws = xTx.I * (xMat.T *yMat)
    return ws


def regression0(ws, xArr):
    y = xArr*ws
    return y

def verifyresults():
    xarr, yarr = loadDataSet('ex0.txt')
    xmat = mat(xarr)
    ymat = mat(yarr)
    ws = standRegression(xarr, yarr)
    predictResult = (ws.T*xmat.T)
    similarity = corrcoef(predictResult, ymat)
    return similarity[0][1]


def plotRegression():
    xarr, yarr = loadDataSet('ex0.txt')
    xmat = mat(xarr)
    ymat = mat(yarr)
    ws = standRegression(xarr, yarr)
    yhat = xmat * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xmat[:, 1].flatten().A[0], ymat.T[:, 0].flatten().A[0])
    xcopy = xmat.copy()
    xcopy.sort(0)
    yhat = xcopy * ws
    ax.plot(xcopy[:, 1],yhat)
    plt.show()


def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy


def ridgRegres(xMat, yMat, lam = 0.2):
    yMean = mean(yMat, 0); xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    yMat = yMat - yMean
    xMat = (xMat - xMeans) /xVar
    xTx = xMat.T * xMat
    demon = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(demon) == 0.0:
        print("This matrix is singular, cannot find inverser")
        return
    ws = xTx.I * (xMat.T *yMat)
    return ws

#####################################################################################################

def drawlwlr(k=1.0):
    xar, yar = loadDataSet('ex0.txt')
    yhat = lwlrTest(xar, xar, yar, k)
    xmat = mat(xar)
    ymat = mat(yar)
    srtInd = xmat[:, 1].argsort(0)
    xSort = xmat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yhat[srtInd])
    ax.scatter(xmat[:, 1].flatten().A[0], ymat.T[:, 0].flatten().A[0], s=2, c='red')
    plt.show()

drawlwlr(0.015)
