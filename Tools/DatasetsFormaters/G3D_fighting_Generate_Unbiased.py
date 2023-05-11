import random

import numpy as np

from Tools import DataSetReader
from Tools.Gesture.MorphologyGetter import MorphologyGetter


def doExtractNewG3DData():
    sequencesOfFighting = [22,23,24,43,44,45,64,65,66,85,86,87,127,128,129,148,149,150,169,170,171,190,191,192,214,215,
                           216,106,107,108]


    pathLabel = "C:\workspace2\Datasets\G3D\Label\\"
    pathData = "C:\workspace2\Datasets\G3D\Data\\"
    pathSplit = "C:\workspace2\Datasets\G3D\Split\\"


    random.seed(3)
    random.shuffle(sequencesOfFighting)
    train = sequencesOfFighting[:26]
    test = sequencesOfFighting[26:]
    print("train : ",train, " len : ",len(train))
    print("test : ",test, " len : ",len(test))
    device = "kinectV1"
    # each label is a sequence in the same order : [1, 2, 3, 4, 5]
    # the labels are
    # 1;PunchRight
    # 2;PunchLeft
    # 3;KickRight
    # 4;KickLeft
    # 5;Defend

    # We will create new sequences using :
    # - variable number of gestures per sequence by splitting the sequence in two at variable index
    # - variable order by mirroring the sequence

    #mirroring index changement : 1->2, 2->1, 3->4, 4->3, 5->5
    mapMirroring = {1:2, 2:1, 3:4, 4:3, 5:5}

    testDatasToSet = []

    for seq in test:
        labFile = open(pathLabel + str(seq), "r")
        lines = labFile.readlines()
        labFile.close()

        labels = []
        for l in lines:
            if len(l.split(",")) == 4:
                classid,start,end,actionPoint = l.split(",")
                labels.append((int(classid),int(start),int(end),int(actionPoint)))

        whereToSplit = random.randint(0,len(labels)-1)
        firstLabels = labels[:whereToSplit]
        secondLabels = labels[whereToSplit:]

        # get the corresponding data
        dataFile = open(pathData + str(seq), "r")
        lines = dataFile.readlines()
        dataFile.close()

        if len(firstLabels) != 0:
            whereToSplitOnData = (firstLabels[-1][2] + secondLabels[0][1]) // 2
        else:
            whereToSplitOnData = 0

        firstData = lines[:whereToSplitOnData]
        secondData = lines[whereToSplitOnData:]

        doReverseSeq1 = random.randint(0,1) == 1 and len(firstLabels) != 0
        doReverseSeq2 = random.randint(0,1) == 1

        if doReverseSeq1:
            firstLabels = [(mapMirroring[classid],start,end,actionPoint) for classid,start,end,actionPoint in firstLabels]
            #reverse data, format of each line is x y z x y z x y z ..., reversing become -x y z -x y z -x y z ...
            for i in range(len(firstData)):
                firstData[i] = " ".join([str(-float(x)) if j%3 == 0 and float(x)!=0 else x for j,x in enumerate(firstData[i].split(" "))])
            firstData = np.array([firstData[i].split(" ") for i in range(len(firstData))], dtype=np.float32)  # -> [time,nbjoint*3]
            firstData = firstData.reshape([firstData.shape[0], firstData.shape[1] // 3, 3])  # -> [time,nbjoint,3]
            mapping = [MorphologyGetter.getMirrorMember(device, i) for i in range(firstData.shape[1])]
            firstData = firstData[:, mapping, :]  # -> [time,nbjoint,3], mirrored
            firstData = firstData.reshape([firstData.shape[0], firstData.shape[1] * 3])  # -> [time,nbjoint*3]
            firstData = [" ".join([str(x) for x in firstData[i]])+"\n" for i in range(len(firstData))]
        secondLabels = [(classid, start - whereToSplitOnData,
                        end - whereToSplitOnData,actionPoint - whereToSplitOnData) for classid,start,end,actionPoint in secondLabels]

        if doReverseSeq2:
            secondLabels = [(mapMirroring[classid],start,end,actionPoint) for classid,start,end,actionPoint in secondLabels]
            #reverse data, format of each line is x y z x y z x y z ..., reversing become -x y z -x y z -x y z ...
            for i in range(len(secondData)):
                secondData[i] = " ".join([str(-float(x)) if j%3 == 0 and float(x)!=0 else x for j,x in enumerate(secondData[i].split(" "))])
            secondData = np.array([secondData[i].split(" ") for i in range(len(secondData))],
                                 dtype=np.float32)  # -> [time,nbjoint*3]
            secondData = secondData.reshape([secondData.shape[0], secondData.shape[1] // 3, 3])  # -> [time,nbjoint,3]
            mapping = [MorphologyGetter.getMirrorMember(device, i) for i in range(secondData.shape[1])]
            secondData = secondData[:, mapping, :]  # -> [time,nbjoint,3], mirrored
            secondData = secondData.reshape([secondData.shape[0], secondData.shape[1] * 3])  # -> [time,nbjoint*3]
            secondData = [" ".join([str(x) for x in secondData[i]])+"\n" for i in range(len(secondData))]
        print("---------------------------------------")

        print("seq : ",seq, " whereToSplit : ",whereToSplit, " whereToSplitOnData : ",whereToSplitOnData,)
        print("doReverseSeq1 : ",doReverseSeq1, " doReverseSeq2 : ",doReverseSeq2)
        print("firstLabels : ",firstLabels)
        print("secondLabels : ",secondLabels)
        print("---------------------------------------")

        #write the new data
        if len(firstLabels) != 0:
            newSeq = "test_modif"+str(seq)
            f = open(pathData + newSeq, "w")
            f.write("".join(firstData))
            f.close()

            f = open(pathLabel + newSeq, "w")
            f.write("\n".join([str(classid) + "," + str(start) + "," + str(end) + "," + str(actionPoint) for classid,start,end,actionPoint in firstLabels]))
            f.close()
            testDatasToSet.append(newSeq)

        newSeq2 ="test2_modif"+str(seq)
        f = open(pathData + newSeq2, "w")
        f.write("".join(secondData))
        f.close()

        f = open(pathLabel + newSeq2, "w")
        f.write("\n".join([str(classid) + "," + str(start) + "," + str(end) + "," + str(actionPoint) for
                           classid, start, end, actionPoint in secondLabels]))

        f.close()
        testDatasToSet.append(newSeq2)


    #write the new split
    f = open(pathSplit + "splitFighting_unbiased.txt", "w")
    f.write("train \n")
    f.write(",".join([str(x) for x in train]))
    f.write("\n")
    f.write("test \n")
    f.write(",".join([str(x) for x in testDatasToSet]))

if __name__ == "__main__":
    doExtractNewG3DData()