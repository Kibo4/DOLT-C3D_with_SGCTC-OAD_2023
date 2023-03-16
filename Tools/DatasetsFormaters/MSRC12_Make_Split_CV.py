"""
Transform the full dataset to the sub dataset eaxch category is separated
This is the protocol used in papers where is category is considered
-> export the split in a txt file, for each fold
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
pathFilesAllC = "C:\\workspace2\\Datasets\\MSRC12\\IdentifiersCategoriesCX\\"
pathDB = "C:\\workspace2\\Datasets\\MSRC12\\"
filesCX = ["allC1.txt", "allC2.txt", "allC3.txt", "allC4.txt", "allC5.txt"]

pathSplitOut = "C:\\workspace2\\Datasets\\MSRC12\\Split\\"
files = os.listdir(dataSetPath)

for fileCX in filesCX:
    f = open(pathFilesAllC + fileCX, "r")
    filenames: List[str] = f.readlines()[0].split(",")  # on the first line, all filenames split by coma ,
    f.close()

    filenamesToKeep: List[str] = []
    seqByPerson: Dict[int, List[str]] = {}

    extractClass = lambda s: int(s.split("_")[2].split("A")[0])
    extractPersonne = lambda s: int(s.split("p")[1])

    # separate each sequence by person
    for file in filenames:
        classe = extractClass(file) + 1  # +1 because we let 0 for the "nothing" class
        filenamesToKeep.append(file)
        pers = extractPersonne(file)
        if pers not in seqByPerson:
            seqByPerson[pers] = []
        seqByPerson[pers].append(file)

    print("nb person", len(seqByPerson))
    nbFold = 10

    folds: List[Tuple[List[str], List[str]]] = []
    # print("newcount ", len(filenamesToKeep))

    list_pers_listOfFile: List[Tuple[int, List[str]]] = list(seqByPerson.items())
    # print("list_pers_listOfFile ", list_pers_listOfFile)
    cpt = 0
    random.seed(2)
    # build the fold
    # the constraint is to have each category in each test fold
    for fileToCopt in range(nbFold):

        todos = [e for e in range(1,13)]  # 12 actions
        train: List[str] = []
        tests: List[str] = []
        shuffle(list_pers_listOfFile)  # change the order
        # print("list_pers_listOfFile ", list_pers_listOfFile)
        for pers, listOfFile in list_pers_listOfFile:

            # if we have already all the categories in the test set, then put the rest in the train set
            if len(todos) == 0:
                for f in listOfFile:
                    train.append(f)
                continue

            classesOfThisPers = list(map(lambda s: extractClass(s), listOfFile))
            found = False
            # search if this person contains a category that is not in the test set
            for cl in classesOfThisPers:
                if cl in todos:
                    found = True
                    todos.remove(cl)
            # if we have at least one category that is not in the test set,
            # then put all the sequence of this person in the test set
            # else put all the sequence of this person in the train set
            if (found):
                for f in listOfFile:
                    tests.append(f)  # add all classes of a person
            else:
                for f in listOfFile:
                    train.append(f)
        # print("todos ", todos)
        assert len(todos) == 0
        # print("len(train)", len(train))
        # print("len(tests)", len(tests))
        assert len(train) + len(tests) == len(filenamesToKeep)
        folds.append((train, tests))


    # test to check if no persons of training is in testing
    for training, testing in folds:
        personnesInTesting = [extractPersonne(e) for e in testing]
        personnesInTraining = [extractPersonne(e) for e in training]
        # print("---\nfold")
        # print("personnesInTesting", personnesInTesting)
        # print("personnesInTraining", personnesInTraining)
        for t in personnesInTraining:
            assert t not in personnesInTesting

    print("Example")
    print("train", folds[0][0], "...")
    print("len train", len(folds[0][0]))

    print("test", folds[0][1][:7])

    if (not os.path.exists(pathSplitOut)):
        os.mkdir(pathSplitOut)
    f = open(pathDB+"db.info","r")
    infos = eval(f.read())
    f.close()
    ## print in file
    for id, (training, testing) in enumerate(folds):
        protocol = "training\n"
        protocol += ",".join(map(lambda f: f + ".txt", training)) + "\n"
        protocol += "testing\n"
        protocol += ",".join(map(lambda f: f + ".txt", testing))
        nameSplit =  fileCX[3:5]+"_fold" + str(id)
        f = open(pathSplitOut + nameSplit+ ".txt", "w+")
        f.write(protocol)
        f.close()

        infos["protocol"] = nameSplit+".txt"
        f = open(pathDB + "db"+nameSplit+".info", "w+")
        f.write(str(infos))
        f.close()



    print("done", fileCX)
print("all done")
