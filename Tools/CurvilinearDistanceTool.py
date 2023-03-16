from typing import List, Tuple

from Tools.Gesture.LabeledSequence import LabeledSequence
import numpy as np


def extractLabelPerFrame_ClassAndWindow(sequence: LabeledSequence, numberPosturePerSegment:List[int],retPerSegmLab=False) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param sequence:
    :param numberPosturePerSegment: cudi frames associations (called sometime "repeat")
    :return: a tuple labels (CuDi space), labels (temporal space)
        each tuple contained 2 elements :
        - window : the delta-frame corresponding to the diffrence between the index of begin frame of the current action,
            and the index of the current frame. IT IS NOT USED ANYMORE IN OUR METHOD. could be removed.
        -  class : the class of the current frame
        # - classSequencePadded : the sequence of classes in the sequence not associated with farme
    """
    sortedLabels = sorted(sequence.labels, key=lambda l: l.beginPostureId)
    labelsFinal = np.zeros([len(numberPosturePerSegment),2],dtype=np.int32)# window, class, classSequencePadded
    labelClasseTemporel = np.zeros([sum(numberPosturePerSegment),2],dtype=np.int32)# window, class, classSequencePadded
    cumulativeCuDiFrame:np.ndarray[int] = np.cumsum(numberPosturePerSegment) - 1

    classStartEndSegment = []
    # print("sorted ", sortedLabels)
    for idLabel, label in enumerate(sortedLabels):
        begin = label.beginPostureId
        end = label.endPostureId
        classe = label.classId

        #find the begin in CuDi space
        distTobegin = abs(cumulativeCuDiFrame-begin)
        startCuDiIndex = np.argmin(distTobegin)

        #find the end in CuDi space
        distToEnd = abs(cumulativeCuDiFrame - end)
        endCuDiIndex = np.argmin(distToEnd)

        #CuDiLabel
        for i in range(startCuDiIndex,endCuDiIndex+1):
            labelsFinal[i] = [i-startCuDiIndex,classe]

        #TEmporalLabel
        for i in range(begin, end + 1):
            if(i<len(labelClasseTemporel)):
                labelClasseTemporel[i] = [i-begin,classe]
            else:
                print("label over the lenght of sequence : len:",len(labelClasseTemporel)," end : ",end)
        classStartEndSegment.append([classe,startCuDiIndex,endCuDiIndex])

    if retPerSegmLab:
        return labelsFinal,labelClasseTemporel,classStartEndSegment
    return labelsFinal,labelClasseTemporel
