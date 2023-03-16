from typing import List, Tuple, Dict

from Tools.Gesture.Joint import Joint
from Tools.Gesture.Label import Label
from Tools.Gesture.Morphology import Morphology
from Tools.Gesture.Posture import Posture

class FormatStringSkeletonException(Exception):
    def __init__(self,msg):
        super(FormatStringSkeletonException, self).__init__(msg)

printed = False
def readOneLine(line:str, morphology:Morphology , nbSkeleton:int)->Tuple[Posture,Posture]:
    nbJointPerSkeleton = morphology.nbJoints
    jointsTypes = morphology.jointTypes
    centerize = False
    # print('line',line)
    values = line.split()
    if len(values) != nbJointPerSkeleton * nbSkeleton * 3:

        if nbSkeleton ==1 and len(values) == 2 * nbJointPerSkeleton * 3 :
            global printed
            if not printed:
                print("warning, two skeletons given while only one is expected, KEEP THE FIRST ONE")
                printed = True
            values = values[:nbJointPerSkeleton * 3]
        else:
            print("NOT ", nbJointPerSkeleton, "*", nbSkeleton, "*3 values but " + str(len(values)))
            raise FormatStringSkeletonException("Attendu : "+str(nbSkeleton)+"skeleton * "+str(nbJointPerSkeleton)+
                                                " joints * 3 = "+str(nbJointPerSkeleton * nbSkeleton * 3)+" \n"+
                                            str(len(values))+" valeurs reçues ")
    joints = []

    values1 = values[:(nbJointPerSkeleton * 3)]  # look only at the firsxt skeleton
    for valId in range(0, len(values1) - 2, 3):
        x = float(values1[valId])
        y = float(values1[valId + 1])
        z = float(values1[valId + 2])
        joint = Joint((x, y, z), jointsTypes[valId // 3])
        joints.append(joint)
    post1 = Posture(joints, morphology, normalize=centerize)

    if nbSkeleton==1:
        return post1, None

    joints2 = []
    values2 = values[(25 * 3):]  # look only at the second skeleton
    for valId in range(0, len(values2) - 2, 3):
        x = float(values2[valId])
        y = float(values2[valId + 1])
        z = float(values2[valId + 2])
        joint = Joint((x, y, z), jointsTypes[valId // 3])
        joints2.append(joint)
    post2 = Posture(joints2, morphology, normalize=centerize)
    return post1, post2


def readDataPostures(fData: str, nbSkeleton: int, morphology: Morphology, SubSampling: int = 1) -> Tuple[
    List[Posture], List[Posture]]:
    """

    :param fData: the path of the file formated
    :param nbSkeleton: assert nbSkeleton==1 or nbSkeleton==2
    :param nbJointPerSkeleton: 20 for kinect v1, 25 for v2...
    :param SubSampling:int - sample 1 frame from every "SubSampling" frames for these videos.
    :return:
    """
    assert nbSkeleton == 1 or nbSkeleton == 2
    fData = open(fData, "r")
    postures1 = []
    postures2 = [] if nbSkeleton == 2 else None
    cpt = 0

    for line in fData:
        if cpt % SubSampling != 0:  # sample 1 frame from every 4 frames for these videos.
            cpt += 1
            continue
        post1, post2 = readOneLine(line,morphology,nbSkeleton)

        postures1.append(post1)

        if nbSkeleton == 1:
            cpt += 1
            continue
        postures2.append(post2)
        cpt += 1
    fData.close()
    return postures1, postures2


def readLabels(fData: str, actionsNames:List[str] , SubSampling: int = 1) -> List[Label]:
    """
    :param fData : The path of the formated label
    :param actionsNames:
    :param fData: the path of the data (include the fileName)
    :return: a list of labels
    """
    fData = open(fData, "r")
    labels = []
    for line in fData:
        values = line.split(",")
        if len(values) <3 :
            print("NOT 3 values but " + str(len(values)))
        classId = int(values[0])
        begin = round(int(values[1]) / SubSampling)
        end = round(int(values[2]) / SubSampling)
        lab = Label(actionsNames[classId] if len(actionsNames)>classId else "UNDEFINED", classId, begin,
                    end)  # they index their actions beginning by '1'
        labels.append(lab)
    fData.close()
    return labels
