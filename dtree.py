import math
from numpy import *
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels


def calcShannonEnt(dataset):
    numEntries = len(dataset)
    lableConts = {}
    for iter in dataset:
        currentlable = iter[-1]     #assuming last one is the lable
        if currentlable not in lableConts.keys():
            lableConts[currentlable] = 0
        lableConts[currentlable] += 1
    shannonEnt = 0.0
    entlist = {}
    for key in lableConts:
        prob = float(lableConts[key])/numEntries
        entlist[key] = prob * math.log(prob, 2)
    entvals = list(entlist.values())
    shannonEnt = array(entvals).sum()
    return -shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataset): #assume the last one is lable
    baseEnt = calcShannonEnt(dataset)
    #print("BaseEntro %f" % baseEnt)
    numFetures = len(dataset[0]) -1
    bestInfoGain = 0.0; bestfeature = -1
    totals = len(dataset)
    for i in range(numFetures):
        featureList = [example[i] for example in dataset]
        uniqset = set(featureList)
        newEntropy = 0.0
        for val in uniqset:
            newset = splitDataSet(dataset,i,val)
            prob = len(newset) / totals
            newEntropy += prob*calcShannonEnt(newset)
            ratios = -1*prob*math.log(prob,2)
        infogain = (baseEnt - newEntropy)/ratios
        if infogain > bestInfoGain:
            bestInfoGain=infogain; bestfeature =i
        #print("for %i axis, InfoGain = %f" % (i,infogain))
    return bestfeature,bestInfoGain

def domajoritycnt(classlist):
    classcount = {}
    for vote in classlist:
        if vote not in classlist.keys() :
            classcount[vote] =0
        classcount[vote] +=1
    sortedclasscount = sorted(classcount.iteritems(), key=operator.intemgetter(1), reverse=True)
    return sortedclasscount[0][0]

def createtree(dataset, orilables):
    lables = orilables
    classlist = [example[-1] for example in dataset]
    if len(set(classlist)) == 1:
        return classlist[0]
    datasetwithoutlable = [d[:-1] for d in dataset]
    uniquemat = []
    for iter in datasetwithoutlable:
        if iter not in uniquemat:
            uniquemat.append(iter)
    if len(dataset[0]) ==1 or len(uniquemat) == 1:
        return domajoritycnt(dataset)
    bestfeat,infogain = chooseBestFeatureToSplit(dataset)
    bestfeatlable = lables[bestfeat]
    myTree = {bestfeatlable:{}}
    del(lables[bestfeat])
    featValues = set([example[bestfeat] for example in dataset])
    for vals in featValues:
        sublables = lables[:]
        myTree[bestfeatlable][vals] = createtree(splitDataSet(dataset,bestfeat,vals),sublables)
    return  myTree

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename,'wb+')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import  pickle
    fr = open(filename,'rb+')
    return pickle.load(fr)

