"""
Script to convert the MAD dataset from .mat to .txt
"""
import os

import numpy as np
import h5py
from typing import IO, List

from Tools.DatasetsFormaters.PosturesToFile import PostureToFile
from Tools.Gesture.Joint import Joint
from Tools.Gesture.MorphologyGetter import MorphologyGetter
from Tools.Gesture.Posture import Posture

pathDB_In = "D:\Datasets\BrutDataMAD\Sub_all\\"

pathData = "C:\\workspace2\\Datasets\\MAD\\Data\\"
pathLabel = "C:\\workspace2\\Datasets\\MAD\\Label\\"

subFolders = os.listdir(pathDB_In)

def extractData(sk):
    frames = []
    data = h5py.File(sk)
    for v in data["skeleton"]:
        for a in v:
            frames.append(a)
    postures: List[Posture] = []
    for f in frames:
        postureArray = np.transpose(data.get(f))#[20, 3]
        morph = MorphologyGetter.kinectV1Morphology()
        joints = [Joint(tuple(e), morph.jointTypes[i]) for i, e in enumerate(postureArray)]
        p = Posture(joints, morph)
        postures.append(p)
    return postures


def extractLabel(lab):
    label = h5py.File(lab)
    labs = label["label"].value

    # %%
    labs = np.transpose(labs)
    labelsFinal = []
    for classId, begin, start in labs[:, :3]:
        if classId >= 1000:
            continue
        labelsFinal.append((classId, begin, start))
    return labelsFinal


def doTreat(path):
    if(not os.path.exists(pathData)):
        os.mkdir(pathData)
    if (not os.path.exists(pathLabel)):
        os.mkdir(pathLabel)
    for f in os.listdir(path):
        #sorry for duplication....
        sk1 = path+f+"/seq01_sk.mat"
        lab1 = path+f+"/seq01_label.mat"

        sk2 = path+f+"/seq02_sk.mat"
        lab2 = path+f+"/seq02_label.mat"

        skel1 = extractData(sk1)
        label1 = extractLabel(lab1)

        skel2 = extractData(sk2)
        label2 = extractLabel(lab2)

        name1= f+"_01"
        name2= f+"_02"

        PostureToFile.toFile(skel1,pathData+name1)
        PostureToFile.toFile(skel2,pathData+name2)

        labsStr1 = [",".join(map(lambda l :str(int(l)),lab)) for lab in label1]
        allLabels1 = "\n".join(labsStr1)

        labsStr2 = [",".join(map(lambda l :str(int(l)),lab)) for lab in label2]
        allLabels2 = "\n".join(labsStr2)


        fData = open(pathLabel + name1, "w+")
        fData.write(allLabels1)
        fData.close()

        fData = open(pathLabel + name2, "w+")
        fData.write(allLabels2)
        fData.close()


doTreat(pathDB_In)
