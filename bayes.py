from numpy import *


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    # 1 is abusive, 0 not
    return postingList,classVec


def createvocablist(dataset):
    vocabset = set([])
    for document in dataset:
        vocabset = vocabset | set(document)
    return list(vocabset)


def setofwords2vec(vocablist, inputset):
    returnvec = [0] *len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] = 1
        else: print("the word %s is not in my vocabulary.." % word)
    return returnvec


def loadtrainset():
    dataset, lables = loadDataSet()
    vocablist = createvocablist(dataset)
    trainmat =[]
    for doc in dataset:
        vec = setofwords2vec(vocablist,doc)
        trainmat.append(vec)
    return mat(trainmat), lables, vocablist


def trainNBO(trainmatrix,trainCategory):
    probLable ={}
    uniqlables = set(trainCategory)
    samplenum = len(trainCategory)
    probLableAttributes = []
    for item in uniqlables:
        icount = trainCategory.count(item)
        probLable[item] = icount/samplenum
 #   for i in range(len(trainmatrix)):
 #       probLableAttributes[list(probLable.keys()).index(trainCategory[i])] += trainmatrix[i]
    return probLable,probLableAttributes


