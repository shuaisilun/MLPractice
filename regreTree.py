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


def binSplitDataSet(dataset, feat, value):
    mat0 = dataset[nonzero(dataset[:, feat] > value)[0], :][0]
    mat1 = dataset[nonzero(dataset[:, feat] <= value)[0], :][0]
    return mat0, mat1


def regleaf(dataset):
    return  mean(dataset[:, -1])


def regErr(dataset):
    return var(dataset[:, -1]) * shape(dataset)[0]


def choosebestsplit(dataset, leaftype, errtype, ops):
    tols = ops[0] ; tolN = ops[1]
    if len(set(dataset[:, -1].T.tolist()[0])) == 1 : return None, leaftype(dataset)
    m, n = shape(dataset)
    s = errtype(dataset)
    bests = inf ; bestindex = 0; bestvalue = 0
    for featindex in range(n-1):
        for splitvalue in set(dataset[:, featindex]):
            mat0, mat1 = binSplitDataSet(dataset, featindex, splitvalue)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errtype(mat0) + errtype(mat1)
            if newS < bests:
                bestindex = featindex
                bestvalue = splitvalue
                bests = newS
    if (s - bests) < tols:
        return None, leaftype(dataset)
    mat0, mat1 = binSplitDataSet(dataset, bestindex, bestvalue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leaftype(dataset)
    return bestindex, bestvalue

def creatTree(dataset, leaftype = regleaf, errtype = regErr, ops =(1,4)):
    feat, val = choosebestsplit(dataset, leaftype, errtype, ops)
    if feat == None : return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lset, rset = binSplitDataSet(dataset, feat, val)
    retTree['left'] = creatTree(lset, leaftype, errtype, ops)
    retTree['right'] = creatTree(rset, leaftype, errtype, ops)
    return  retTree


def isTree(obj):
    return  type(obj).__name__ == ('dict')


def getMean(tree):
    if isTree(tree['right']) : tree['right'] = getMean(tree['right'])
    if isTree(tree['left']) : tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    if shape(testData)[0] == 0 : return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lset, rset = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']) : tree['left'] = prune(tree['left'], lset)
    if isTree(tree['right']) : tree['right'] = prune(tree['right'], rset)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lset, rset = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lset[:,-1] - tree['left'], 2)) + sum(power(rset[:, -1] - tree['right'], 2))
        treemean = (tree['left'] + tree['right']) /2.0
        erroMerge = sum(power(testData[:, -1] - treemean, 2))
        if erroMerge < errorNoMerge:
            return treemean
        else:
            return tree
    else:
        return tree


def linearSovle(dataset):
    m, n = shape(dataset)
    x = mat(ones((m,n))) ; y = mat(ones(shape = (m,1)))
    x[:, 1:n] = dataset[:, 0:n-1]; y = dataset[:,-1]
    xTx = x.T * x
    if linalg.det(xTx) == 0:
        raise  NameError("This matrix is singular cant find inverse")
    ws = xTx.I * (x.T * y)
    return ws, x, y


def modelLeaf(dataset):
    ws, X, Y = linearSovle(dataset)
    return ws


def modelErr(dataset):
    ws, x, y = linearSovle(dataset)
    yhat = x * ws
    return sum(power(y - yhat, 2))


def testcode():
    mydata = loadDataSet('exp2.txt')
    mymat = mat(mydata)
    regreTree= creatTree(mymat, leaftype= modelLeaf, errtype= modelErr, ops=(1, 10))
    #testdata = mat(loadDataSet('ex2test.txt'))
    #prune(regreTree, testdata)
    print regreTree


testcode()
