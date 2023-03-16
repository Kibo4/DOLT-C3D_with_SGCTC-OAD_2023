"""
This script is used to convert the Chalearn dataset from .mat to .txt
"""

import os

import scipy.io
from typing import IO

pathTrain = "D:\\workspace2\\Datasets\\Chalearn\\DataBrutes\\trainingMat\\"
pathValidation = "D:\\workspace2\\Datasets\\Chalearn\\DataBrutes\\validationMat\\"
pathTest = "D:\\workspace2\\Datasets\\Chalearn\\DataBrutes\\testMat\\"

pathData = "D:\\workspace2\\Datasets\\Chalearn\\Data\\"
pathLabel = "D:\\workspace2\\Datasets\\Chalearn\\Label\\"
pathActions = "D:\\workspace2\\Datasets\\Chalearn\\Split\\Actions.csv"

pathSplit = "D:\\workspace2\\Datasets\\Chalearn\\Split\\Split.txt"

trainFiles = os.listdir(pathTrain)
validFiles = os.listdir(pathValidation)
testFiles = os.listdir(pathTest)


def _readActions(file: IO):
    actions = []
    for line in file:
        className = line.split(';')[1].rstrip()  # format : id;class name\n
        actions.append(className)
    return actions


f = open(pathActions, "r")
actions = _readActions(f)
f.close()


def doTreat(listFile, path):
    for f in listFile:
        data = scipy.io.loadmat(path + f)

        fileDest = f[:-4] + ".txt"  # remove ".mat" and add ".txt"

        numFrame = data["Video"]["NumFrames"][0, 0][0][0]
        fullData = ""
        for i in range(numFrame):
            theFrame = ""
            for joint in range(20):
                pos = data["Video"]["Frames"][0, 0][0, i][0]['WorldPosition'][0, 0][joint]
                theFrame += " ".join(map(lambda x: str(x), pos)) + " "
            fullData += theFrame[:-1] + "\n"  # to not have the last comma

        fData = open(pathData + fileDest, "w+")
        fData.write(fullData)
        fData.close()

        numLabels = data["Video"]["Labels"][0, 0][0].shape[0]
        allLabels = ""
        for i in range(numLabels):
            labelName = data["Video"]["Labels"][0, 0][0][i][0][0]
            begin = data["Video"]["Labels"][0, 0][0][i][1][0][0]
            end = data["Video"]["Labels"][0, 0][0][i][2][0][0]
            labelId = str(actions.index(labelName))
            label = labelId + ',' + str(begin) + ',' + str(end)
            allLabels += label + "\n"

        fData = open(pathLabel + fileDest, "w+")
        fData.write(allLabels)
        fData.close()


doTreat(trainFiles, pathTrain)
doTreat(validFiles, pathValidation)
# doTreat(testFiles,pathTest)

trainFiles = map(lambda x: str(x[:-4] + ".txt"), trainFiles)
validFiles = map(lambda x: str(x[:-4] + ".txt"), validFiles)
# testFiles = map(lambda x: str(x[:-4] + ".txt"), testFiles) # test are not annotated

# not usable
# fSplit = open(pathSplit,"w+")
# fSplit.write("Train : "+"\n")
# fSplit.write(",".join(trainFiles)+"\n")
# fSplit.write("Valid : "+"\n")
# fSplit.write(",".join(validFiles)+"\n")
# fSplit.write("Test : "+"\n")
# fSplit.write(",".join(testFiles)+"\n")
# fSplit.close()
