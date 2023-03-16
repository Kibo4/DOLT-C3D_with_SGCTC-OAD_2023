"""
Converts the OAD dataset to the standard format
"""
import os

import scipy.io
from typing import IO

pathData = "D:\\workspace2\\Datasets\\OAD\\DataBrutes\\"

pathDataSubData = "skeleton\\"
pathDataSubLabel = "label\\label.txt"

pathActions = "D:\\workspace2\\Datasets\\OAD\\Split\\Actions.csv"

pathDataOut = "D:\\workspace2\\Datasets\\OAD\\Data\\"
pathLabelOut = "D:\\workspace2\\Datasets\\OAD\\Label\\"

files = os.listdir(pathData)


def _readActions(file: IO) -> []:
    actions = []
    for line in file:
        className = line.split(';')[1].rstrip()  # format : id;class name\n
        actions.append(className)
    return actions


f = open(pathActions, "r")
actions = _readActions(f)
f.close()


def doTreatOne(listFileFrame, path):
    fullLinesToExport = []
    firstNb = None
    for f in listFileFrame:  # for each frame (1file = 1frame)
        file = open(path + f, "r")
        lines = file.readlines()[:25]
        file.close()
        if len(lines) < 2:  # skip blank files, skeleton not capted at the begin
            continue
        if firstNb ==None:
            firstNb = int(f[:-4])-1

        lines = list(map(lambda l : l.strip(),filter(lambda l: l.strip() != "", lines)))
        # [ "j1.x j1.y j1.z", "j2.x j2.y,j2.z"..]
        if  len(lines) != 25:
            print("erreur fichier"+path+f)
            assert False # 25 joints
        fullLine = " ".join(lines)

        fullLinesToExport.append(fullLine)

    return "\n".join(fullLinesToExport),firstNb


def doTreatOneLabel(pathlabel,negOffset):
    file = open(pathlabel, "r")
    lines = file.readlines()
    file.close()
    lines = list(filter(lambda x: x.strip() != "", lines))
    if len(lines) < 2:  # skip blank files, skeleton not capted at the begin
        print("Erreur " + pathlabel + " nb lignes <2")
        assert False


    outputLines = []

    actionId = 0
    for l in lines:  # for each frame (1file = 1frame)
        if l[0].isalpha():
            actionId = actions.index(l.lower().strip())
            continue
        begin, end = l.split()
        begin = str(int(begin)-negOffset)
        end = str(int(end)-negOffset)
        outputLines.append(str(actionId)+","+begin+","+end)

    return "\n".join(outputLines)


sequencesFolderName = os.listdir(pathData)
id = 0
for seq in sequencesFolderName:
    theFolder = pathData + seq + "\\" + pathDataSubData
    theFileLabel = pathData + seq + "\\" + pathDataSubLabel
    if id%10==0:
        print(id,"/",len(sequencesFolderName))
    # first order by name, then by len -> natural order
    listFiles = sorted(sorted(os.listdir(theFolder)), key=lambda name: len(name))
    theSeqFileTxt,negOffset = doTreatOne(listFiles, theFolder)
    fileOut = open(pathDataOut + seq + ".txt", "w+")
    fileOut.write(theSeqFileTxt)
    fileOut.close()

    theSeqLabel = doTreatOneLabel(theFileLabel,negOffset)

    fileOut = open(pathLabelOut + seq + ".txt", "w+")
    fileOut.write(theSeqLabel)
    fileOut.close()
    id+=1
print("DONE")


