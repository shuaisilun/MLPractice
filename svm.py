from numpy import *
from time import sleep


def loadDataSet(fileName):
    dataMat = []; labelmat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelmat.append(float(lineArr[2]))
    return dataMat,labelmat


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


def smoSimple(datamatIn, labels, C, toler, maxIter):
    datamat = mat(datamatIn); labelmat = mat(lables).T
    m, n = datamat.shape
    b = 0; alphas = mat(zeros((m,1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(multiply(alphas, labelmat).T * (datamat * datamat[i, :].T)) + b
            Ei = fxi - float(labelmat[i])
            if ((labelmat[i]*Ei < -toler and alphas[i] < C) or (lablemat[i]*Ei > toler and alphas[i]>0)):
                j = selectJrand(i,m)
                fxj = float(multiply(alphas,labelmat).T * (datamat * datamat[j, :].T)) + b
                Ej = fxj - float(labelmat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelmat[i] == lablemat[j]):
                    H = min(C, alphas[j] + alphas[i])
                    L = max(0, alphas[j] + alphas[i] - C)
                else:
                    H = min(C, alphas[j] - alphas[i] + C)
                    L = max(0, alphas[j] - alphas[i])
                if L == H : print ("L==H"); continue
                eta = 2.0 * datamat[i,:] * datamat[j,:].T - datamat[i, :]*datamat[i, :].T - datamat[j, :]*datamat[j, :].T
                if eta >= 0 : print("eta >=0"); continue
                alphas[j] -= labelmat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print("J not moving enough"); continue
                alphas[i] += labelmat[j]*lablemat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - labelmat[i] * (alphas[i] - alphaIold) * datamat[i, :] * datamat[i, :].T - lablemat[j] * (alphas[j] - alphaJold) * datamat[i, :] * datamat[j, :].T
                b2 = b - Ej - labelmat[i] * (alphas[i] - alphaIold) * datamat[i, :] * datamat[j, :].T - lablemat[j] * (alphas[j] - alphaJold) * datamat[j, :] * datamat[j, :].T
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


def testsmosimple():
   d, l = loadDataSet('testSet.txt')
   b, alphas = smoSimple(d, l, 0.6, 0.001, 40)
   return b, alphas


def kernelTrans(X, A, kTup):
    m,n = shape(X)
    k = mat(zeros((m,1)))
    if kTup[0] == 'lin': k = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            k[j] = deltaRow * deltaRow.T
        k = exp(k / (-1.0*kTup[1]**2))
    else: raise NameError("kernel function not recognized")
    return k


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)
        #for i in range(self.m):
        #    for j in range(self.m):
        #        self.K[i,j] = float(kernelTrans(self.X[i], self.X[j], kTup))


def calcEk(os, k):
    fxk = float(multiply(os.alphas, os.labelMat).T * os.K[:, k] + os.b)
    Ek = fxk - float(os.labelMat[k])
    return Ek


def selectJ(i, os, Ei):
    maxK = -1 ; maxDeltaE = 0; Ej =0
    os.eCache[i] = [1,Ei]
    validEcacheList = nonzero(os.eCache[:,0].A)[0]
    if (len(validEcacheList)) >1 :
        for k in validEcacheList:
            if k == i : continue
            Ek = calcEk(os, k)
            deltaE = abs(Ei -Ek)                                #select the most progressive second alpha
            if (deltaE > maxDeltaE):
                maxDeltaE = deltaE; maxK = k ; Ej = Ek
        return  maxK, Ej
    else:
        j = selectJrand(i, os.m)
        Ej = calcEk(os, j)
    return j, Ej


def updateEK(os,k):
    Ek = calcEk(os, k)
    os.eCache[k] = [1, Ek]


def innerL(i, os):
    Ei = calcEk(os, i )
    if ((os.labelMat[i]*Ei < -os.tol) and (os.alphas[i] < os.C) or ((os.labelMat[i]*Ei > os.tol) and (os.alphas[i] >0))):
        j,Ej = selectJ(i, os, Ei)
        alphaIold = os.alphas[i].copy() ; alphaJold = os.alphas[j].copy()
        if (os.labelMat[i] != os.labelMat[j]):
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L == H: print("L==H"); return 0
        eta = 2.0 * os.K[i, j] - os.K[i, i] - os.K[j, j]
        if eta >= 0 : print("eta>=0"); return 0
        os.alphas[j] -= os.labelMat[j] * (Ei - Ej)/eta
        os.alphas[j] = clipAlpha(os.alphas[j], H, L)
        updateEK(os, j)
        if (abs(os.alphas[j] - alphaJold) < 0.00001):
            print("J not moving enough"); return 0
        os.alphas[i] += os.labelMat[j]*os.labelMat[i]*(alphaJold- os.alphas[j])
        updateEK(os, i)
        b1 = os.b - Ei - os.labelMat[i]*(os.alphas[i] - alphaIold)* os.K[i, i] - os.labelMat[j]*(os.alphas[j] - alphaJold)*os.K[i, j]
        b2 = os.b - Ej - os.labelMat[i]*(os.alphas[i] - alphaIold)* os.K[i, j] - os.labelMat[j]*(os.alphas[j] - alphaJold)*os.K[j, j]
        if (0 < os.alphas[i]) and (os.C > os.alphas[i]): os.b = b1
        elif (0 < os.alphas[j]) and (os.C > os.alphas[j]) : os.b = b2
        else: os.b = (b1+b2)/2.0
        return 1
    else: return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, KTup = ('lin',0)):
    os = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, KTup)
    iter = 0
    entireSet = True ; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged >0)  or entireSet):
        alphaPairsChanged =0
        if entireSet:
            for i in range(os.m):
                alphaPairsChanged += innerL(i, os)
            print("fullset, iter:%d i:%d pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, os)
                print("none-bound, iter: %d, i:%i ,pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False
        elif (alphaPairsChanged == 0) : entireSet = True
        print("iteration number %d " % iter)
    return os.b, os.alphas


def testsmop(Ktup):
    d, l = loadDataSet('testSet.txt')
    if Ktup == 'lin':
        b, alphas = smoP(d, l, 0.6, 0.001, 40)
    elif Ktup == 'rbf':
        b, alphas = smoP(d, l, 200, 0.0001,10000, ('rbf', 1.3))
    return b, alphas

def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))