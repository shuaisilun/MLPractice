import  sys
from numpy import  mat, mean, power

def readInput(file):
    for line in file:
        yield line.rstrip()


input = readInput(sys.stdin)
input = [float(line) for line in input]
numInputs = len(input)
input = mat(input)
sqInput = power(input, 2)

print("%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput)))
#print("report: still alive")