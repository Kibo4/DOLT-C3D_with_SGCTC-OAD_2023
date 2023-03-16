"""
Transform the full dataset to the sub dataset containing only the C4 modality and the 6 Iconic actions
Used in Boulahia RFIAP. And Bloom for some papers.
"""

import os
import random
import shutil
from random import shuffle
from copy import deepcopy
from itertools import groupby
from typing import List, Tuple, Dict


dataSetPath = "C:\\workspace2\\Datasets\\MSRC12\\Data\\"
dataSetLabel = "C:\\workspace2\\Datasets\\MSRC12\\Label\\"
pathFilesC4 = "C:\\workspace2\\Datasets\\MSRC12\\Split\\allC4.txt"

dataSetPathOut = "C:\\workspace2\\Datasets\\MSRC6_IconicC4\\Data\\"
dataSetLabelOut = "C:\\workspace2\\Datasets\\MSRC6_IconicC4\\Label\\"

pathSplitToFill = "C:\\workspace2\\Datasets\\MSRC6_IconicC4\\Split\\"

files = os.listdir(dataSetPath)

actionsToKeepId = [2,4,6,8,10,12] # Duck, Google, Shoot, Throw, Change weapon, Kick
# I did the Actions.csv by hand


# read the file which contains all C4 modality file
f = open(pathFilesC4,"r")
filenames:List[str] = f.readlines()[0].split(",") # on the first line, all filenames split by coma ,
f.close()

# create the table List[List[str]] which contains [classes,sequences] to have sequences by classes
filenamesToKeep:List[str] = []
seqByPerson:Dict[int,List[str]] = {}

extractClass = lambda s : int(s.split("_")[2].split("A")[0])
extractPersonne = lambda s : int(s.split("p")[1])

for file in filenames:
    classe = extractClass(file)

    try:
        newActionIdless1 = actionsToKeepId.index(classe) # the new action id -1 (because we let 0 for the "nothing" class)
    except ValueError: # if it's not one of the selected actions to keep
        continue
    filenamesToKeep.append(file)
    pers = extractPersonne(file)
    if pers not in seqByPerson:
        seqByPerson[pers] = []
    seqByPerson[pers].append(file)


print("nb person",len(seqByPerson))
nbFold = 10

folds:List[Tuple[List[str],List[str]]] = []
print("newcount ",len(filenamesToKeep))

list_pers_listOfFile:List[Tuple[int,List[str]]] = list(seqByPerson.items())
print("list_pers_listOfFile ",list_pers_listOfFile)
cpt = 0
random.seed(2)
for fileToCopt in range(nbFold):

    todos = [e for e in actionsToKeepId]
    train: List[str] = []
    tests: List[str] = []
    shuffle(list_pers_listOfFile) # change the order
    print("list_pers_listOfFile ", list_pers_listOfFile)
    for pers, listOfFile in list_pers_listOfFile:
        if(len(todos)==0):
            for f in listOfFile:
                train.append(f)
            continue

        classesOfThisPers = list(map(lambda s:extractClass(s), listOfFile))
        found = False
        for cl in classesOfThisPers:
            if cl in todos:
                found = True
                todos.remove(cl)
        if(found):
            for f in listOfFile:
                tests.append(f) # add all classes of a person
        else:
            for f in listOfFile:
                train.append(f)
            continue

    folds.append((train,tests))
    print("len(train)",len(train))
    print("len(tests)",len(tests))

    assert  len(train)+len(tests) == len(filenamesToKeep)



# test to check if no persons of training is in testing
for training,testing in folds:
    personnesInTesting = [extractPersonne(e) for e in testing]
    personnesInTraining = [extractPersonne(e) for e in training]
    print("---\nfold")
    print("personnesInTesting",personnesInTesting)
    print("personnesInTraining",personnesInTraining)
    for t in personnesInTraining:
        assert t not in personnesInTesting


print("Example")
print("train",folds[0][0],"...")
print("len train",len(folds[0][0]))

print("test",folds[0][1][:7])

if(not os.path.exists(pathSplitToFill)):
    os.mkdir(pathSplitToFill)
## print in file
for id,(training,testing) in enumerate(folds):
    protocol = "training\n"
    protocol += ",".join(map(lambda f:f+".txt",training))+"\n"
    protocol += "testing\n"
    protocol += ",".join(map(lambda f:f+".txt",testing))
    f = open(pathSplitToFill+"fold"+str(id)+".txt","w+")
    f.write(protocol)
    f.close()

#copy the data
if(not os.path.exists(dataSetPathOut)):
    os.mkdir(dataSetPathOut)
for fileToCopt in filenamesToKeep:
    shutil.copy(dataSetPath + fileToCopt+".txt",dataSetPathOut+fileToCopt+".txt")

# translate and copy the labels
if(not os.path.exists(dataSetLabelOut)):
    os.mkdir(dataSetLabelOut)
for fileToCopt in filenamesToKeep:
    f = open(dataSetLabel+fileToCopt+".txt","r")
    lines = f.readlines()
    f.close()
    for id,line in enumerate(lines):
        split = line.split(",")
        classe = int(split[0])
        try:
            newIndex =  actionsToKeepId.index(classe)+1
        except ValueError as ve:
            print("SHOULD NEVER HAPPEN, a file kept have a label ",classe," and we keep only ",actionsToKeepId,
                  " ",fileToCopt)
            raise ve
        split[0] = str(newIndex)
        newLine = ",".join(split)
        lines[id] = newLine
    f = open(dataSetLabelOut + fileToCopt + ".txt", "w+")
    f.writelines(lines)
    f.close()
