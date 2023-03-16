from typing import List, Tuple

import numpy as np



def getId_Start_ActionPoints_End(pathLabelOriginal: str, file: str) -> List[Tuple[int, int, int, int]]:
    """
    Must be called only if the sequence have action points ! otherwise raise error
    :param file: the
    :param pathLabelOriginal: the path to find the original label description per sequence ( without seq file name)
    :raise IndexOutOfBound if the sequence does not have action point
    :return: the list of actions which occurs in the file with the tuple (actionClassId, actionStart,actionPoint,ActionEnd)
    """
    f = open(pathLabelOriginal + file.replace(".npy", ""), "r")
    actionId_ActionPoint = filter(lambda l: l != "", f.readlines())  # non empty lines
    actionId_ActionPoint = list(map(lambda line: (int(line.split(",")[0]),  # id class
                                                  int(line.split(",")[1]),  # start
                                                  -1 if len(line.split(","))<4 else int(line.split(",")[3]),  # action point
                                                  int(line.split(",")[2])), actionId_ActionPoint))  # end
    f.close()
    return actionId_ActionPoint

def getCuDiSplit(pathCuDiSplit,file:str)->List[int]:
    file = ".".join(file.split(".")[:-1])
    f = open(pathCuDiSplit+file,"r")
    splitCuDi = list(map(int, f.readlines()[0].split(",")))
    f.close()
    return splitCuDi

def BoundedActionTemporalIoUMetric(file: str, boundsPrediction, pathLabelOriginal: str, nbClass: int,canCorrect,
                                   minCouverture:float):
    """

    # :param pathCuDiSplit: the path to find the CuDi details (number of temporal frames per CuDi frames)
    # :param pathLabelOriginal: the path to find the original label description per sequence ( without seq file name)
    # :param nbFrameBefore: the nubmer of frame before the action point to Evaluate
    # :param stategy: the strategy to decide, *restitute in temporal domain*
    :param boundsPrediction: boundsPrediction
    :param file: the name of the sequence
    # :param predictions: list[int] (CuDi space)
    # :param rejection: List[0<float<1] represent the confidence, per prediction
    :param GT: the ground truth per CuDi frame
    :param nbClass: the number of classes
    :param TOLERANCE: in number of frame the tolerance allowed to detect an action before it starts,
                should depends of quality of annotation
    :return:Tp,FP,FN,resultsDetailed.
                resultsDetailed[number of actionPoints,nbFrameBefore+1] :
                        -5 : the corresponding frame is before the action start
                        -4 : the corresponding frame is not an action (0)
                        -3 : no CuDi frame in this ratio
                        -2 : FP or FN
                        -1 : no starts found between action start and observed frame
                        0  : Reject prediction
                        1  : TP
    """

    # assert len(predictions) == len(cuDiSplit)
    # -1 to match with temporal index
    actionsId_Points = getId_Start_ActionPoints_End(pathLabelOriginal, file)

    #classe,start, end
    boundsGT: List[Tuple[int, int, int]] = list(map(lambda classe_start_AP_end: (classe_start_AP_end[0],classe_start_AP_end[1],classe_start_AP_end[3]),actionsId_Points ))

    # strategy must emit 0 action class bound


    TP,FP,MatConf,nbPerclasses,Precision_c,Recall_c,earliness,NLToD_c, \
    TPAll,FPAll,Precision,Recall,NLToD,nbActionsTotalGT,\
    TrueAcceptAt,FalseAcceptAt,detailedResult = BoundedActionTemporalIoUMetric_apply(boundsPrediction,boundsGT,
                                                                                     canCorrect,minCouverture,nbClass)

    return TP,FP,MatConf,nbPerclasses,Precision_c,Recall_c,earliness,NLToD_c, \
    TPAll,FPAll,Precision,Recall,NLToD,nbActionsTotalGT,TrueAcceptAt,FalseAcceptAt,detailedResult




def IoU_deb(start, end, startGT, endGT):
    """

    :param start: pred
    :param end:  pred
    :param startGT: GT
    :param endGT: GT
    :return: Intersection over Union, considering union from start and not from startGT
    we dont want to penalize the non-earliness on this criteria
    """

    maxiStart = max(start,startGT)
    miniEnd = min(end,endGT)

    if miniEnd < maxiStart: #no overlap
        return 0
    intersection = miniEnd-maxiStart+1

    areaPred = (end-start)+1
    assert max(startGT,start)<=endGT
    areaGTTroncated = (endGT-max(startGT,start))+1 # troncated with start of prediction


    union = areaPred+areaGTTroncated-intersection
    return intersection/union
#     maxiStart = max(start,startGT)
#     miniEnd = min(end,endGT)
#
#


def BoundedActionTemporalIoUMetric_apply(boundsPredictions:List[Tuple[int, int, int]] ,boundsGTTemporal:List[Tuple[int, int, int]] ,canCorrect:bool
             ,minCouverture:float, nbClass:int):
    """


    :param boundsPredictions: List[class id,start, end], temporal domain
    :param boundsGTTemporal: List[class,start, end], temporal domain
    :param canCorrect: true if allow correction by the system, if the first detection was a False Positive
    :param minCouverture: the minimum couverture to be match the GT
    :param nbClass: the number of class
    :return:
    detailedResult : 1 :TP
                     0 : reject
                    -1 : FP on action (wrong class)
                    -2 : FP on action (IoU <Mini)
                    -3 : FP on BG/flag taken
    """

    boundsPredictions.sort(key=lambda classe_start_end:classe_start_end[1]) # in place
    # boundsGTTemporal = boundsGTTemporal.numpy().tolist()
    boundsGTTemporal.sort(key=lambda classe_start_end:classe_start_end[1]) # in place


    flags = [0]*len(boundsGTTemporal)

    TP = np.zeros([nbClass+1])
    FP = np.zeros([nbClass+1])
    MatConf = np.zeros([nbClass+1,nbClass+1]) # +1: non-action,
    earliness = [[] for _ in range(nbClass+1)]
    nbPerclasses = np.zeros([nbClass+1])
    TrueAcceptAt = []
    FalseAcceptAt = []

    detailedResult = []

    for idGT, (classIdGT,startGT, endGT) in enumerate(boundsGTTemporal):
        nbPerclasses[classIdGT] += 1

    for idPrediction,(classId,start,end)in enumerate(boundsPredictions):
        classId = int(classId)
        # get the GT which match the best the given prediction
        gtIndex = int(np.argmax(list(map(lambda clID_startlab_endlab : IoU_deb(start,end,clID_startlab_endlab[1],clID_startlab_endlab[2]), boundsGTTemporal))))
        #get the corresponding value
        gtMax = np.max(list(map(lambda clID_startlab_endlab : IoU_deb(start,end,clID_startlab_endlab[1],clID_startlab_endlab[2]), boundsGTTemporal)))
        classIdGT,startGT,endGT = boundsGTTemporal[gtIndex]
        if flags[gtIndex]==0 and classId==classIdGT and IoU_deb(start,end,startGT,endGT)>minCouverture: # equivalent to gtMax>minCouverture
            flags[gtIndex] = 1
            TP[classIdGT] += 1
            MatConf[classIdGT][classIdGT] += 1

            #length for NDToD
            # nltodVal = min(1,max(0,(start-startGT+1))  /(endGT-startGT+1))
            nltodVal = min(1,max(0,(start-startGT))  /(endGT-startGT+1))

            earliness[classIdGT].append(  nltodVal )
            TrueAcceptAt += [nltodVal]
            detailedResult.append(1)
        else:

            FP[classId] += 1
            MatConf[classIdGT if gtMax>0 else 0][classId] += 1

            if (classId!=classIdGT):
                detailedResult.append(-1)
                        # print("class != ",classId,classIdGT)
            elif flags[gtIndex]==1:
                detailedResult.append(-3)
            else: # print("IOU < ",IoU_deb(start,end,startGT,endGT))
                detailedResult.append(-2)

            if not canCorrect and gtMax>0:
                print("enter in flag to 1, in FP",gtMax)
                flags[gtIndex] = 1
            # nltodVal = min(1,max(0,(start-startGT+1))  /(endGT-startGT+1))
            if gtMax>0:
                nltodVal = min(1,max(0,(start-startGT+1))  /(endGT-startGT+1))
                FalseAcceptAt += [nltodVal]


        #
        # if not found:
        #     FP[classId] += 1
        #     MatConf[-1][classId] += 1
        #     print("FP segmentPred",idPrediction,"on",len(boundsPredictions))
        #     detailedResult.append(-3)


    #--Per class
    Precision_c = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=TP + FP != 0)
    Recall_c = np.divide(TP, nbPerclasses, out=np.zeros_like(TP), where=nbPerclasses != 0)
    NLToD_c = np.array([np.average(elem) for elem in earliness])
    #-- Total, micro average
    TPAll = np.sum(TP)
    FPAll = np.sum(FP)
    Precision = np.divide(TPAll, TPAll + FPAll, out=np.zeros_like(TPAll), where=TPAll + FPAll != 0)
    Recall = TPAll / (len(boundsGTTemporal))
    flat_earliness = [item for sublist in earliness for item in sublist]
    NLToD = np.sum(np.array(flat_earliness))/len(np.array(flat_earliness)) if  len(np.array(flat_earliness))!=0 else 0

    return TP,FP,MatConf,nbPerclasses,Precision_c,Recall_c,earliness,NLToD_c, \
           TPAll,FPAll,Precision,Recall,NLToD,len(boundsGTTemporal),TrueAcceptAt,FalseAcceptAt, detailedResult
