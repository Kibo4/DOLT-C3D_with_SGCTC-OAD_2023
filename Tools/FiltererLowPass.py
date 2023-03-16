from typing import List

from Tools.Gesture.Joint import Joint
from scipy import signal
import numpy as np

from Tools.Gesture.Posture import Posture

sos = signal.butter(3, 10, 'lp', fs=70, output='sos')
def FilterThisSequence(postures:List[Posture]):
    """
    filter the postures in place
    sos = signal.butter(3, 10, 'lp', fs=70, output='sos')
    this respect the online constraint because the filtered vlaues only depends on the previous values
    ex :
    variationsX = [1,20,-1]
    signal.sosfilt(sos, variationsX)
        Out[24]: array([0.04399324, 1.0666792 , 4.02405874])
    variationsX = [1,20,-1,99]
    signal.sosfilt(sos, variationsX)
        Out[26]: array([ 0.04399324,  1.0666792 ,  4.02405874, 11.12767341]) #same firsts values
    :param postures: the sequence to filter
    """
    joints: List[List[Joint]] = list(map(lambda p: p.joints, postures))
    jointsPos = [[] for _ in range(len(postures))]  # len, 20, 3

    for i, listeJ in enumerate(joints):
        liste = [list(j.position) for j in listeJ]
        jointsPos[i] = liste

    zeroFrame = 0
    # find the first non-0 frame position
    for pos in postures:
        if (pos.joints[0].position[0] != 0):
            break
        zeroFrame += 1

    jointsPos = np.array(jointsPos)
    for jId in range(len(jointsPos[0])):
        variationX = jointsPos[:, jId, 0]
        variationY = jointsPos[:, jId, 1]
        variationZ = jointsPos[:, jId, 2]
        filteredX = signal.sosfilt(sos, variationX)
        filteredY = signal.sosfilt(sos, variationY)
        filteredZ = signal.sosfilt(sos, variationZ)

        # to avoid the peak at the beginning, keep the oritginal signal for the
        # 20 first frames
        filteredX = np.concatenate((variationX[:18 + zeroFrame], filteredX[18 + zeroFrame:]))
        filteredY = np.concatenate((variationY[:18 + zeroFrame], filteredY[18 + zeroFrame:]))
        filteredZ = np.concatenate((variationZ[:18 + zeroFrame], filteredZ[18 + zeroFrame:]))

        filteredX = np.expand_dims(filteredX, -1)
        filteredY = np.expand_dims(filteredY, -1)
        filteredZ = np.expand_dims(filteredZ, -1)

        filteredXYZ = np.concatenate((filteredX, filteredY, filteredZ), axis=1)

        assert len(filteredXYZ) == len(jointsPos)
        for idPost in range(len(jointsPos)):
            joints[idPost][jId].position = filteredXYZ[idPost]