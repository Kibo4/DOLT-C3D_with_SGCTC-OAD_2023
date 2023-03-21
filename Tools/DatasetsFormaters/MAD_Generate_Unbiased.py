import random

from Tools import DataSetReader


def doExtractnewMADData():
    trainSeq = ["sub15_01","sub18_01","sub19_01","sub02_01","sub03_01","sub16_01","sub11_01","sub04_01","sub13_01",
                 "sub08_01","sub14_01","sub01_01","sub17_01","sub12_01","sub09_01","sub20_01","sub15_02","sub18_02",
                 "sub19_02","sub02_02","sub03_02","sub16_02","sub11_02","sub04_02","sub13_02","sub08_02","sub14_02",
                 "sub01_02","sub17_02","sub12_02","sub09_02","sub20_02"]
    testSeq = ["sub05_01","sub10_01","sub07_01","sub06_01",
                 "sub05_02","sub10_02","sub07_02","sub06_02"]

    pathLabel = "C:\workspace2\Datasets\MAD\Label\\"
    pathData = "C:\workspace2\Datasets\MAD\Data\\"
    pathSplit = "C:\workspace2\Datasets\MAD\Split\\"



    # each label is a sequence in the same order : [1, 2, 3, 4, 5]
    # the labels are
    # 1;Running
    # 2;Crouching
    # 3;Jumping
    # 4;Walking
    # 5;Jump and Side-Kick
    # 6;Left Arm Swipe to the Left
    # 7;Left Arm Swipe to the Right
    # 8;Left Arm Wave
    # 9;Left Arm Punch
    # 10;Left Arm Dribble
    # 11;Left Arm Pointing to the Ceiling
    # 12;Left Arm Throw
    # 13;Swing from Left (baseball swing)
    # 14;Left Arm Receive
    # 15;Left Arm Back Receive
    # 16;Left Leg Kick to the Front
    # 17;Left Leg Kick to the Left
    # 18;Right Arm Swipe to the Left
    # 19;Right Arm Swipe to the Right
    # 20;Right Arm Wave
    # 21;Right Arm Punch
    # 22;Right Arm Dribble
    # 23;Right Arm Pointing to the Ceiling
    # 24;Right Arm Throw
    # 25;Swing from Right (baseball swing)
    # 26;Right Arm Receive
    # 27;Right Arm Back Receive
    # 28;Right Leg Kick to the Front
    # 29;Right Leg Kick to the Right
    # 30;Cross Arms in the Chest
    # 31;Basketball Shooting
    # 32;Both Arms Pointing to the Screen
    # 33;Both Arms Pointing to Both Sides
    # 34;Both Arms Pointing to Right Side
    # 35;Both Arms Pointing to Left Side

    # We will create new sequences using :
    # - variable number of gestures per sequence by splitting the sequence in two at variable index
    # - variable order by mirroring the sequence

    #mirroring index changement based on left/right, if not mirrored, the index is the same
    mapMirroring = {1:1, 2:2, 3:3, 4:4, 5:5, 6:19, 7:18, 8:20, 9:21, 10:22, 11:23, 12:24, 13:25, 14:26, 15:27, 16:29,
                    17:28, 18:6, 19:7, 20:8, 21:9, 22:10, 23:11, 24:12, 25:13, 26:14, 27:15, 28:17, 29:16, 30:30, 31:31,
                    32:32, 33:33, 34:34, 35:35}

    # mirror of Running is Running
    # mirror of Crouching is Crouching
    # mirror of Jumping is Jumping
    # mirror of Walking is Walking
    # mirror of Jump and Side-Kick is Jump and Side-Kick
    # mirror of Left Arm Swipe to the Left is Right Arm Swipe to the Right
    # mirror of Left Arm Swipe to the Right is Right Arm Swipe to the Left
    # mirror of Left Arm Wave is Right Arm Wave
    # mirror of Left Arm Punch is Right Arm Punch
    # mirror of Left Arm Dribble is Right Arm Dribble
    # mirror of Left Arm Pointing to the Ceiling is Right Arm Pointing to the Ceiling
    # mirror of Left Arm Throw is Right Arm Throw
    # mirror of Swing from Left (baseball swing) is Swing from Right (baseball swing)
    # mirror of Left Arm Receive is Right Arm Receive
    # mirror of Left Arm Back Receive is Right Arm Back Receive
    # mirror of Left Leg Kick to the Front is Right Leg Kick to the Front
    # mirror of Left Leg Kick to the Left is Right Leg Kick to the Right
    # mirror of Right Arm Swipe to the Left is Left Arm Swipe to the Right
    # mirror of Right Arm Swipe to the Right is Left Arm Swipe to the Left
    # mirror of Right Arm Wave is Left Arm Wave
    # mirror of Right Arm Punch is Left Arm Punch
    # mirror of Right Arm Dribble is Left Arm Dribble
    #...

    random.seed(1)
    testDatasToSet = []
    for seq in testSeq:
        labFile = open(pathLabel + str(seq), "r")
        lines = labFile.readlines()
        labFile.close()

        labels = []
        for l in lines:
            if len(l.split(",")) == 3:
                classid,start,end = l.split(",")
                labels.append((int(classid),int(start),int(end)))

        numberOfSplit = random.randint(1,10)
        print("Splitting " + str(seq) + " in " + str(numberOfSplit) + " sequences")


        whereToSplit = len(labels) // (numberOfSplit )
        print("Splitting at each " + str(whereToSplit) + " labels")
        labelsSplitted = [labels[i:i + whereToSplit] for i in range(0, len(labels), whereToSplit)]
        print("labelsSplitted : " + str(labelsSplitted) )



        # get the corresponding data
        dataFile = open(pathData + str(seq), "r")
        lines = dataFile.readlines()
        dataFile.close()

        whereToSplit = [0]
        datas = []
        for i in range(len(labelsSplitted)):

            if i == len(labelsSplitted)-1:
                whereToSplitOnData = len(lines)-1
            else:
                whereToSplitOnData = (labelsSplitted[i][-1][2] + labelsSplitted[i+1][0][1]) // 2

            data = lines[whereToSplit[-1]:whereToSplitOnData]

            doReverseSeq = random.randint(0,1) == 1

            if doReverseSeq:
                labelsSplitted[i] = [(mapMirroring[classid],start-whereToSplit[-1],end-whereToSplit[-1]) for classid,start,end in labelsSplitted[i]]
                #reverse data, format of each line is x y z x y z x y z ..., reversing become -x y z -x y z -x y z ...
                for idata in range(len(data)):
                    data[idata] = " ".join([str(-float(x)) if j%3 == 0 and float(x)!=0 else x for j,x in enumerate(data[idata].split(" "))])
            else:
                labelsSplitted[i] = [(classid,start-whereToSplit[-1],end-whereToSplit[-1]) for classid,start,end in labelsSplitted[i]]
            datas.append(data)
            whereToSplit.append(whereToSplitOnData)


            print("---------------------------------------")

            print("seq : ",seq, " whereToSplit : ",whereToSplit, " whereToSplitOnData : ",whereToSplitOnData,)
            print("doReverseSeq1 : ",doReverseSeq)
            print("labels : ",labelsSplitted)
            print("i : ",i)
            print("labels : ",labelsSplitted[i])
            print("---------------------------------------")

            #write the new sequences
            newSeq ="test"+str(i)+"_modif"+str(seq)
            print("writing newSeq : ",newSeq)
            f = open(pathData + newSeq, "w")
            f.write("".join(data))
            f.close()

            f = open(pathLabel + newSeq, "w")
            f.write("\n".join([str(classid) + "," + str(start) + "," + str(end) for
                               classid, start, end in labelsSplitted[i]]))

            f.close()
            testDatasToSet.append(newSeq)


    #write the new split
    f = open(pathSplit + "split_unbiased.txt", "w")
    f.write("train \n")
    f.write(",".join([str(x) for x in trainSeq]))
    f.write("\n")
    f.write("test \n")
    f.write(",".join([str(x) for x in testDatasToSet]))

if __name__ == "__main__":
    doExtractnewMADData()