"""
This file is used to format the G3D dataset into a format usable by the project.
ie : the standard format used in PKU-MMD dataset, which is a txt file containing the 3D coordinates of each joint
of each frame.
"""

import os
import random
import xml.etree.ElementTree as ET
from copy import deepcopy
from random import shuffle
from typing import List, Tuple, Dict

from Tools.DatasetsFormaters.PosturesToFile import PostureToFile
from Tools.Gesture.Joint import Joint
from Tools.Gesture.JointType import JointType
from Tools.Gesture.MorphologyGetter import MorphologyGetter
from Tools.Gesture.Posture import Posture


def parseTagFile(pathFile):
    tree = ET.parse(pathFile)
    root = tree.getroot()
    listOfTag: List[Tuple[str, int, int]] = []
    for tags in root.findall("Tag"):
        action = tags[0].text.strip()
        start = int(tags[1].text.strip())
        end = int(tags[2].text.strip())
        listOfTag.append((action, start, end))
    return listOfTag


def parseActionPointFile(pathFile):
    tree = ET.parse(pathFile)
    root = tree.getroot()
    listOfTag: List[Tuple[str, int]] = []
    for tags in root.findall("ActionPoint"):
        action = tags[0].text.strip()
        AP = int(tags[1].text.strip())
        listOfTag.append((action, AP))
    return listOfTag


def parsePostureFile(pathFile) -> List[Joint]:
    tree = ET.parse(pathFile)
    root = tree.getroot()
    joints = list(root.iter("Joint"))

    jointsTypes: List[JointType] = MorphologyGetter.kinectV1Morphology().jointTypes

    if (len(joints) == 0):
        return [Joint((0, 0, 0), jt) for jt in jointsTypes]
    assert len(joints) == 20
    listeOfPos: List[Joint] = []
    for idJT, joint in enumerate(joints):
        position = joint[0]
        x = float(position[0].text)
        y = float(position[1].text)
        z = float(position[2].text)
        listeOfPos.append(Joint((x, y, z), jointsTypes[idJT]))
    return listeOfPos


def actionID(action: str, ACTION: List[str]):
    for id, act in enumerate(ACTION):
        if action == act:
            return id
    raise Exception('ACTION ' + action + " NOT FOUND")


def doLabel(pathTag, pathActionPoint, num, pathOutLabel, ACTIONS: List[str], mapping, isDriving, isFPS, isMisc):
    fileTag = pathTag + "Tags" + num + ".xml"
    fileActionPoint = pathActionPoint + "ActionPoints" + num + ".xml"

    listOftag = parseTagFile(fileTag)

    listOfAP: List[Tuple[str, int]] = parseActionPointFile(fileActionPoint)
    if isFPS or isMisc:
        toRemoveDuplication = ["Walk", "Run", "Climb"]
        if (isMisc):
            toRemoveDuplication = ["Wave", "Flap", "Clap"]
        newListOfAP = deepcopy(listOfAP)
        for ap in range(1, len(listOfAP)):
            action = listOfAP[ap][0]
            if action in toRemoveDuplication:
                if listOfAP[ap - 1][0] == action:
                    newListOfAP.remove(listOfAP[ap])
        listOfAP = newListOfAP
    try:
        assert len(listOfAP) == len(listOftag)
    except Exception as e:
        if isDriving:
            assert len(listOftag) == 1
            listOfAP = listOfAP[:1]
        else:
            print(" len(listOfAP)", len(listOfAP), "len(listOftag)", len(listOftag), "  Action ", num, fileTag)
            raise e
    toPrint = ""
    for i in range(len(listOfAP)):
        act, start, end = listOftag[i]
        act2, AP = listOfAP[i]
        try:
            start = mapping[str(start)]
            end = mapping[str(end)]
            AP = mapping[str(AP)]
        except Exception as e:
            print("action ", num, fileTag)
            raise e
        if not isDriving:
            assert act == act2

        if not os.path.exists(pathOutLabel):
            os.mkdir(pathOutLabel)
        actID = actionID(act, ACTIONS)
        toPrint += str(actID) + "," + str(start) + "," + str(end) + "," + str(AP) + "\n"
    file = open(pathOutLabel + num, "w+")
    file.write(toPrint)
    file.close()


def doData(pathData, num, pathOutData) -> Dict:
    files = os.listdir(pathData)
    mapping = {}  # le mapping num>id dans le fichier, pour retrouver le bon mapping pour les labels et AP
    postures: List[Posture] = []

    files.sort(key=lambda name: int(name.replace("Skeleton ", "").replace(".xml", "")))
    for id, f in enumerate(files):
        indexReal = f.replace("Skeleton ", "").replace(".xml", "")
        mapping[indexReal] = str(id)
        listOfPos = parsePostureFile(pathData + f)
        p: Posture = Posture(listOfPos, MorphologyGetter.kinectV1Morphology())
        postures.append(p)

    if not os.path.exists(pathOutData):
        os.mkdir(pathOutData)
    PostureToFile.toFile(postures, pathOutData + str(num))

    return mapping


def TreatAll(pathData, pathTag, pathActionPoint, pathOutData, pathOutLabel, ACTIONS: List[str]):
    files = os.listdir(pathData)

    for f in files:
        num = f.replace("KinectOutput", "")
        mapping = doData(pathData + f + "/Skeleton/", num, pathOutData)
        doLabel(pathTag, pathActionPoint, num, pathOutLabel, ACTIONS, mapping, "Driving" in pathData, "FPS" in pathData,
                "Misc" in pathData)


def readActionsCSV(pathActionFile):
    f = open(pathActionFile, "r")
    lines = f.readlines()
    f.close()

    return list(map(lambda l: l.split(";")[1].strip(), lines))


ACTIONS = readActionsCSV("C:\workspace2\Datasets\G3D\Actions.csv")  # file done by hand
assert len(ACTIONS) == 21  # 20 clases +nothing

folders = ["Bowling", "Driving", "Fighting", "FPS", "Golf", "Misc", "Tennis"]


# Uncomment to export normal normal data from xml.
# for f in folders:
#     print(f)
#     TreatAll("C:\workspace2\Datasets\G3D\BrutData\\"+f+"\\", "C:\workspace2\Datasets\G3D\BrutData\Tags\\",
#          "C:\workspace2\Datasets\G3D\BrutData\G3DActionPointsv2\\","C:\workspace2\Datasets\G3D\Data\\",
#          "C:\workspace2\Datasets\G3D\Label\\",ACTIONS)
# print("Done with success")


def TreatAllSplits_20_10_HoldOut(pathData: str, pathOutSplit: str, category: str):
    """
    Split the data in 20 seq for train and 10 for test
    :param pathData:
    :param pathOutSplit:
    :param category:
    :return:
    """
    files = os.listdir(pathData)
    numbers = list(map(lambda f: f.replace("KinectOutput", ""), files))
    random.seed(42)
    shuffle(numbers)
    test = numbers[:10]
    train = numbers[10:]  # 20 sequence for train (exept for Bowling which is 19)
    print("test", test, len(test))
    print("train", train, len(train))
    if not os.path.exists(pathOutSplit):
        os.mkdir(pathOutSplit)
    with open(pathOutSplit + "split" + category + "holdout20_10.txt", "w+") as f:
        f.write("train\n")
        f.write(",".join(train))
        f.write("\ntest\n")
        f.write(",".join(test))
        f.close()


def TreatAllSplits_Ssubjects_Xfolds(pathData: str, pathOutSplit: str, category: str, numberOfSubjectOut: int,
                                    numberOfFold: int,
                                    isBowling: bool):
    """
    Split the data in numberOfSubjectOut subject for test and the rest for train.
    :param pathData:
    :param pathOutSplit:
    :param category:
    :param numberOfSubjectOut:
    :param numberOfFold:
    :param isBowling:
    :return:
    """
    files = os.listdir(pathData)
    numbers = list(map(lambda f: f.replace("KinectOutput", ""), files))
    numbers = sorted(numbers, key=lambda n: int(n))
    # 3 consecutives elements are from the same subject
    # except for Bowling which is 2 for the 1th group
    seqPerSubject = []
    if not isBowling:
        seqPerSubject = [numbers[:3], numbers[3:6], numbers[6:9], numbers[9:12], numbers[12:15],
                         numbers[15:18], numbers[18:21], numbers[21:24], numbers[24:27], numbers[27:30]]
    else:
        seqPerSubject = [numbers[:2], numbers[2:5], numbers[5:8], numbers[8:11], numbers[11:14],
                         numbers[14:17], numbers[17:20], numbers[20:23], numbers[23:26], numbers[26:29]]

    if not os.path.exists(pathOutSplit):
        os.mkdir(pathOutSplit)
    idGroups = list(range(10))
    random.seed(42)  # to reproduce the same split
    TakenAsTest = []
    for i in range(numberOfFold):
        shuffle(idGroups)
        testID = idGroups[:numberOfSubjectOut]
        while len(set(testID).intersection(set(TakenAsTest))) > 0:  # theorycally, could be infinite,
            # but it's not for this seed
            shuffle(idGroups)
            testID = idGroups[:numberOfSubjectOut]
        TakenAsTest += testID
        trainID = idGroups[numberOfSubjectOut:]

        train = []
        test = []
        for t in trainID:
            train += seqPerSubject[t]
        for t in testID:
            test += seqPerSubject[t]

        print("test", sorted(test, key=lambda n: int(n)), len(test))
        print("train", sorted(train, key=lambda n: int(n)), len(train))
        with open(pathOutSplit + "split" + category + str(numberOfSubjectOut) + "subjects_out_fold" + str(i) + ".txt",
                  "w+") as f:
            f.write("train\n")
            f.write(",".join(sorted(train, key=lambda n: int(n))))
            f.write("\ntest\n")
            f.write(",".join(sorted(test, key=lambda n: int(n))))
            f.close()


# Export Split files for each categories
print("Exporting splits")
for f in folders:
    print(f)
    TreatAllSplits_Ssubjects_Xfolds("C:\workspace2\Datasets\G3D\BrutData\\" + f + "\\",
                                    "C:\workspace2\Datasets\G3D\Split\\", f, numberOfSubjectOut=1, numberOfFold=10,
                                    isBowling="Bowling" in f)
