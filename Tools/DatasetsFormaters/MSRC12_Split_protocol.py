"""
This script is used to generate the split.txt file for the MSRC12 dataset.
All categories are considered.
Not used in previous papers.
"""
import os
import random
from random import shuffle
from copy import deepcopy
from itertools import groupby

dataSetPath = "C:\\workspace2\\Datasets\\MSRC12\\Data"
pathSplit = "C:\\workspace2\\Datasets\\MSRC12\\Split\\split.txt"

files = os.listdir(dataSetPath)

keyfunc = lambda f:f.split("p")[1].split(".")[0]
files = sorted(files,key=keyfunc)
groups = []
uniquekeys = []
for k, g in groupby(files, keyfunc):
    groups.append(list(g))      # Store group iterator as a list
    uniquekeys.append(k)

keys = deepcopy(uniquekeys)
random.seed(2) # to reproduce the same split
shuffle(keys)
toTakeTest = keys[0:3]
others = keys[3:]

testFiles = []
for ind in toTakeTest:
    take = uniquekeys.index(ind)
    gro = groups[take]
    testFiles+=gro

validFiles = []
for ind in others[0:2]:
    take = uniquekeys.index(ind)
    gro = groups[take]
    validFiles += gro
trainFiles = []
for ind in others[2:]:
    take = uniquekeys.index(ind)
    gro = groups[take]
    trainFiles+=gro

f = open(pathSplit,"w+")
f.write("train\n")
f.write(",".join(trainFiles)+"\n")
f.write("valid\n")
f.write(",".join(validFiles)+"\n")
f.write("test\n")
f.write(",".join(testFiles))
