from numpy import *

def loaddataset(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        currline = line.strip().split('\t')
        fltline = map(float, currline)
        dataMat.append(fltline)
    return dataMat


def distEclud(vectA, vectB):
    return sqrt(sum(power(vectA-vectB,2)))


def randcent(dataset, k):
    n = shape(dataset)[1]
    centroids = mat(zeros(shape = (k,n)))
    for i in range(n):
        minJ = min(dataset[:, i])
        rangeJ = float(max(dataset[:, i])- minJ)
        centroids[:, i] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def kmeans(dataset, k, distmeas = distEclud, createCent = randcent):
    m,n = shape(dataset)
    centroid = createCent(dataset, k)
    changed = True
    clusterAssemt = mat(zeros((m, 2)))
    while changed:
        changed = False
        for i in range(m):
            mindist = inf ; minindex = -1
            for j in range(k):
                dist = distmeas(centroid[j, :], dataset[i, :])
                if dist < mindist:
                    mindist = dist
                    minindex = j
            if clusterAssemt[i, 0] != minindex: changed = True
            clusterAssemt[i, :] = minindex, mindist**2
        #print(centroid)
        for cent in range(k):
            ptsinclust = dataset[nonzero(clusterAssemt[:, 0].A == cent)[0]]
            centroid[cent, :] = mean(ptsinclust, axis=0)
    return centroid, clusterAssemt


def LVQ(dataset, rate):
    labelcount = len(set(dataset[:, -1].A))
    m,n = shape(dataset)
    centroid = randcent(dataset, labelcount)
    loop = 0
    while loop <=500:
        loop = loop + 1
        x = dataset[int(random.rand()* m),:]
        p = 0 ; mindist = inf
        for i in range(labelcount):
            dist = distEclud(x[0:n-1], centroid[i, 0:n-1])
            if dist < mindist:
                mindist = dist
                p = i
        if x[-1] == centroid[p, -1]:
            centroid[p] = centroid[p] + rate * (x - centroid[p])
        else:
            centroid[p] = centroid[p] - rate * (x - centroid[p])
    return centroid




########################################################################
def testcode():
    dmat= mat(loaddataset('testSet.txt'))
    mycentroids, clusterAssing = kmeans(dmat,4)
    print(mycentroids)

testcode()