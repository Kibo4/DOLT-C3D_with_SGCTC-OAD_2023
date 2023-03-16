"""
Transform the raw dataset to the standard format
"""

import math
import os

pathData =       "C:\\workspace2\\Datasets\\MSRC12\\DataBrutes\\MSRC12AnnotForRec\\data\\data\\"
pathTags =       "C:\\workspace2\\Datasets\\MSRC12\\DataBrutes\\MSRC12AnnotForRec\\data\\tagstream\\"
pathFrameLevel = "C:\\workspace2\\Datasets\\MSRC12\\DataBrutes\\MSRC12AnnotForRec\\data\\sepinst\\"

pathDataOUT =       "C:\\workspace2\\Datasets\\MSRC12\\Data\\"
pathLabelOUT =       "C:\\workspace2\\Datasets\\MSRC12\\Label\\"


actions = "C:\\workspace2\\Datasets\\MSRC12\\Actions.csv"



file = open(actions,"r")
Actions = [l.split(';')[1] for l in file.readlines()]
file.close()

if not os.path.isdir(pathDataOUT):
    os.mkdir(pathDataOUT)
if not os.path.isdir(pathLabelOUT):
    os.mkdir(pathLabelOUT)

fileNames = os.listdir(pathData)
filesTag = os.listdir(pathTags)
fileNamesFrameLvl = os.listdir(pathFrameLevel)
withoutExt = [x.split(".")[0] for x in fileNamesFrameLvl]

assert len(fileNames)==len(filesTag)
try:
    assert len(fileNames)==len(fileNamesFrameLvl)
except:
    missed =  [x for x in fileNames if x.split('.')[0] not in withoutExt]
    print("Some exemples are not in sepinst")
    print(missed)

for f in withoutExt:
    print("file "+f)
    file = open(pathData+f+".csv","r")
    allData = file.readlines()
    file.close()
    allDatabyTicks = []
    lastFrameWith0AtBegin = 1
    detectedRealvalues  =False
    for id,line in enumerate(allData):
        vals = line.split()
        try:
            assert len(vals)==81
        except Exception as e:
            print("ERREUR "+f+" line "+str(id))
            raise e
        tick = vals[0]
        vals = vals[1:]
        coordinatesCont = []
        for i in range(0,len(vals),4):
            x,y,z = vals[i],vals[i+1],vals[i+2]
            coordinatesCont += [x,y,z]
            if (float(x)!=0. or float(y)!=0.):
                detectedRealvalues=True
            if not detectedRealvalues and (float(x)==float(y)==float(z)==0.):
                lastFrameWith0AtBegin=id-1
        allDatabyTicks.append((tick,coordinatesCont))


    file = open(pathDataOUT+f+".txt","w+")
    file.write("\n".join([" ".join(x[1]) for x in allDatabyTicks]))
    file.close()


    file = open(pathTags+f+".tagstream")
    allData = file.readlines()
    file.close()
    allData = allData[1:]

    ActionsPointAndClassByActionInFile=[]

    for line in allData:
        splited = line.split(';')
        tick = splited[0]
        actID = int(splited[1].split()[0][1:])
        assert 0 < actID < 13
        tick = (float(tick)*1000+49875./2.)/49875.

        closerDiff = tick-float(allDatabyTicks[0][0])
        closerIndex = 0
        for i in range(1,len(allDatabyTicks)): # can be largely optimised
            diff = math.fabs(tick-float(allDatabyTicks[i][0]))
            if diff<closerDiff:
                closerDiff=diff
                closerIndex=i
        ActionsPointAndClassByActionInFile.append((actID,closerIndex))

    file = open(pathFrameLevel+f+".sep")
    allData = file.readlines()
    file.close()
    try:
        assert len(allData) == len(ActionsPointAndClassByActionInFile)
    except Exception as e:
        print("Frames skiped",lastFrameWith0AtBegin)
        raise e

    for i,line in enumerate(allData):
        start, end = line.split()
        actId,frameIdActionPoint = ActionsPointAndClassByActionInFile[i]
        ActionsPointAndClassByActionInFile[i] = (actId,int(start)+lastFrameWith0AtBegin, int(end)+lastFrameWith0AtBegin,frameIdActionPoint)

    file = open(pathLabelOUT+f+".txt","w+")
    file.writelines("\n".join([",".join([str(e) for e in lab]) for lab in ActionsPointAndClassByActionInFile]))
    file.close()




