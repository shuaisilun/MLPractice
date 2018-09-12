from numpy import *
from time import sleep


def loadDataSet(fileName):
    dataMat = []; lablemat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        lablemat.append(float(lineArr[2]))
    return dataMat,lablemat


def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j


def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(datamatIn,lables, C, toler, maxIter):
    datamat = mat(datamatIn); lablemat = mat(lables).T
    m, n = datamat.shape
    b = 0; alphas = mat(zeros((m,1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(multiply(alphas, lablemat).T * (datamat * datamat[i, :].T)) + b
            Ei = fxi - float(lablemat[i])
            if ((lablemat[i]*Ei < -toler and alphas[i] < C) or (lablemat[i]*Ei > toler and alphas[i]>0)):
                j = selectJrand(i,m)
                fxj = float(multiply(alphas,lablemat).T * (datamat * datamat[j, :].T)) + b
                Ej = fxj - float(lablemat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (lablemat[i] == lablemat[j]):
                    H = min(C, alphas[j] + alphas[i])
                    L = max(0, alphas[j] + alphas[i] - C)
                else:
                    H = min(C, alphas[j] - alphas[i] + C)
                    L = max(0, alphas[j] - alphas[i])
                if L == H : print "L==H"; continue
                eta = 2.0 * datamat[i,:] * datamat[j,:].T - datamat[i, :]*datamat[i, :].T - datamat[j, :]*datamat[j, :].T
                if eta >= 0 : print "eta >=0"; continue
                alphas[j] -= lablemat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "J not moving enough"; continue
                alphas[i] += lablemat[j]*lablemat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - lablemat[i] * (alphas[i] - alphaIold) * datamat[i, :] * datamat[i, :].T - lablemat[j] * (alphas[j] - alphaJold) * datamat[i, :] * datamat[j, :].T
                b2 = b - Ej - lablemat[i] * (alphas[i] - alphaIold) * datamat[i, :] * datamat[j, :].T - lablemat[j] * (alphas[j] - alphaJold) * datamat[j, :] * datamat[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i :%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0) : iter += 1
        else:  iter = 0
        print("iter number %d" % iter)
    return b,alphas



