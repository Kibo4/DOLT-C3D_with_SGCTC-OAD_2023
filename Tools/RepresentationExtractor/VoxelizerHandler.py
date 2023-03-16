from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from Tools.Gesture.Morphology import Morphology
from Tools.Gesture.Posture import Posture


class VoxelizerHandler(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    def isCuDi(self):
        return False

    def isOnlyTwoSkeleton(self):
        return False

    def nbVoxelisation(self):
        return 1

    @abstractmethod
    def finalSizeBox(self) -> Tuple:
        pass

    def getTFType(self):
        return tf.int8

    def getNPType(self):
        return np.int8

    def __init__(self, sizeBoxInit: List[int], thresholdToleranceCuDi: float, threshCurviDist: float,
                 jointSelected: List[int], morphology: Morphology):
        """

        :param sizeBoxInit: the initial size of the 3D image (without joint and skeleton ID) (3items)
        :param thresholdToleranceCuDi: tolerance threshold for consideration into CuDi segment
        :param threshCurviDist: the threshold to reach to fill a segment
        :param thresholdToleranceForVoxelization (aka toleranceDraw): all displacement between two frame below this threshold wont be drawn
        :param jointSelected: the joints which will be taken into account  in the voxelisation process
        """
        self.morphoLogy = morphology
        self.jointSelected: List[int] = jointSelected
        self.tresholdToleranceCuDi = thresholdToleranceCuDi
        self.threshCurviDist = threshCurviDist
        self.sizeBox = sizeBoxInit

    @abstractmethod
    def extractRepresentation(self, data: List[Posture]) -> Tuple[List[np.array], List[int]]:
        """
        :param data:
        :return: - The representation - the number of frame used for each chunk
        """
        pass

    def getTranspositionFeaturesForMirroredVoxelization(self):
        raise NotImplementedError("getMirroredVoxelization not implemented for this voxelizer")

    def getNewListOfPointBetween(self, point1, point2, minDist: float = 1.):
        l: List = [point1]
        current_i = 0
        dist = self.distTo(l[current_i], point2)
        while dist > minDist:
            ratio = minDist / dist
            lastAddedX, lastAddedY, lastAddedZ = l[current_i]
            interP = (lastAddedX + ratio * (point2[0] - lastAddedX),
                      lastAddedY + ratio * (point2[1] - lastAddedY),
                      lastAddedZ + ratio * (point2[2] - lastAddedZ)
                      )
            l.append(interP)
            current_i += 1
            dist = self.distTo(l[current_i], point2)
        l.append(point2)
        return l

    def getListOfPointBetween(self,point1, point2, minDist):
        """
        get the minimun number of point between point1 and point2 included in order that minDist separate each point
        :param point1:
        :param point2:
        :param minDist:
        :return:
        """
        vectDirection = point2 - point1
        normalizedVector = vectDirection / np.linalg.norm(vectDirection)
        nbPoint = int(np.linalg.norm(vectDirection) / minDist)
        l = []
        for i in range(nbPoint+1):
            l.append(point1 + normalizedVector * minDist * i)
        l.append(point2)
        return l


    def distTo(self, p1, p2):
        return np.linalg.norm([p2[i] - p1[i] for i in range(3)])

    def normalize(self, pos: np.array, mini: np.array,
                  maxi: np.array, sizeBox: np.array):
        """
                  0           min -2        formula : ((v-min)/(max-min))*(sizeBox-1)
                  sizeBox     max  15
                  return x,y,z normalized between 0 and sizeBox-1
        """
        if all(maxi - mini == np.zeros_like(mini)):
            return np.zeros_like(mini)
        posNorm = (pos - mini) / (maxi - mini) * (sizeBox - 1)
        posNorm = [min(sizeBox[i] - 1, posNorm[i]) for i in range(len(mini))]
        posNorm = [max(0, posNorm[i]) for i in range(len(mini))]
        return np.array(posNorm)

    def normalizePosture(self,posture, mini, maxi, spines_middle_position, sizeBox, dim=3):
        """
        mini and maxi should be centerized before
        :param posture:
        :param mini:
        :param maxi:
        :param spines_middle_position:
        :param sizeBox:
        :return:
        """
        pCopy = deepcopy(posture)
        for idJ,j in enumerate(posture.joints):
            position = pCopy.joints[idJ].position[:dim] - spines_middle_position[:dim]
            pCopy.joints[idJ].position = self.normalize(position, mini, maxi, np.array(sizeBox))
        return pCopy

