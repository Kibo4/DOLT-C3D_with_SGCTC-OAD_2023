import math
import time
from typing import Tuple, List

import numpy as np

from Tools.Gesture.Morphology import Morphology
from Tools.Gesture.MorphologyGetter import MorphologyGetter
from Tools.Gesture.Posture import Posture
import tensorflow as tf

from Tools.RepresentationExtractor.VoxelizerRCB_3D_Projection2D_1axisXY_JointID_bones_heatmaps_fuzzy2 import \
    MirroirableVoxelization


class VoxelizerRCB_3D_Projection2D_1axisZY_JointID_bones_heatmaps_fuzzy2(MirroirableVoxelization):

    def name(self) -> str:
        return "Project1AxisZY_fuzzy2"

    def finalSizeBox(self) -> Tuple:
        return self.sizeBox[0], self.sizeBox[1], 1 + len(self.jointSelected) + self.morphoLogy.getNbBones(), 1

    def __init__(self, sizeBoxInit, thresholdToleranceCuDi, threshCurviDist,jointSelected, morphology: Morphology):
        super(VoxelizerRCB_3D_Projection2D_1axisZY_JointID_bones_heatmaps_fuzzy2, self).__init__(sizeBoxInit,
                                                                                                 thresholdToleranceCuDi,
                                                                                                 threshCurviDist,
                                                                                                 jointSelected,
                                                                                                 morphology)
        self.nbJoint = len(jointSelected)
        self.transposition = None
        assert self.sizeBox[0] == self.sizeBox[2]," dimX and dimZ must be equals, otherwise it will compromise the " \
                                                  "symmetry and the model usage"

    def isOnlyTwoSkeleton(self):
        return True

    def isCuDi(self):
        return True

    def getNPType(self):
        return np.float32

    def getTFType(self):
        return tf.float32

    def getDistToBoneXY(self, BonePoint1, BonePoint2, i, j):
        """
        get the distance between the point (i,j) and the segment
        :param bone: the point A
        :param bone2: the point B
        :param i: X coordinate
        :param j: Y coordinate
        :return: the distance
        """
        A = BonePoint1[0:2]
        B = BonePoint2[0:2]

        # if are zeros,return max dimension (finalSizeBox)
        if A[0] == 0 and A[1] == 0 and B[0] == 0 and B[1] == 0:
            return self.finalSizeBox()[0]

        if (A[0] == B[0] and A[1] == B[1]):
            return math.sqrt((A[0] - i) ** 2 + (A[1] - j) ** 2)

        AB = B - A
        AC = np.array([i, j]) - A
        BC = np.array([i, j]) - B
        res = None
        if np.dot(AB, AC) < 0:
            res = np.linalg.norm(AC)
        elif np.dot(AB, BC) > 0:
            res = np.linalg.norm(BC)
        else:
            res = np.linalg.norm(np.cross(AB, AC)) / np.linalg.norm(AB)
        # print("res", res)
        return res

    def getDistToBoneZY(self, BonePoint1, BonePoint2, z,y):
        """
        get the distance between the point (0,y,z) and the segment
        ignore X dimension
        :param bone: the point A (x,y,z)
        :param bone2: the point B (x,y,z)
        :param i: Z coordinate
        :param j: Y coordinate
        :return: the distance
        """
        newBonePoint1 = np.array([BonePoint1[2], BonePoint1[1]])
        newBonePoint2 = np.array([BonePoint2[2], BonePoint2[1]])

        return self.getDistToBoneXY(newBonePoint1, newBonePoint2, z, y)

    def nbBones(self):
        return self.morphoLogy.nbBones()

    def extractRepresentation(self, data: List[Posture]) -> Tuple[List[np.array], List[int]]:
        boxes = []
        repeats = []
        postureID = 0
        # start a chrono
        start = time.time()

        rootId = 0
        repeatb4 = 0
        while postureID < len(data) and data[postureID].joints[rootId].position[1] == 0:
            postureID += 1
            repeatb4 += 1

        if postureID == len(data):
            return [np.zeros(shape=[*self.finalSizeBox()], dtype=self.getNPType())], [repeatb4]

        postureID += 2  # we skip the first frame which contain data, to avoid smoothing problem
        repeatb4 += 2
        postureNormalized = data[postureID]
        postureID += 1
        repeatb4 += 1
        headId = self.morphoLogy.getJointIdForName("head")
        def getMinMaxXYZ(postur):
            distHeadRoot = np.linalg.norm(
                np.array(postur.joints[headId].position) - np.array(postur.joints[rootId].position))
            center = postur.joints[0].position
            minPosX = postur.joints[rootId].position[0] - center[0] - distHeadRoot * 0.8
            minPosY = postur.joints[rootId].position[1] - center[1] - distHeadRoot * 1.1
            minPosZ = postur.joints[rootId].position[2] - center[2] + distHeadRoot * 0.3
            maxPosX = postur.joints[rootId].position[0] - center[0] + distHeadRoot * 0.8
            maxPosY = postur.joints[rootId].position[1] - center[1] + distHeadRoot * 1.1
            maxPosZ = postur.joints[rootId].position[2] - center[2] - distHeadRoot * 0.8
            return minPosX, minPosY, minPosZ, maxPosX, maxPosY, maxPosZ, center

        minPosX, minPosY, minPosZ, maxPosX, maxPosY, maxPosZ, center = getMinMaxXYZ(postureNormalized)
        postureNormalized = self.normalizePosture(postureNormalized,
                                                  np.array([minPosX, minPosY,minPosZ]),
                                                  np.array([maxPosX, maxPosY,maxPosZ]), center,
                                                  self.sizeBox)

        # all positions for each joint
        positions = [postureNormalized.joints[i].position for i in self.jointSelected]

        while postureID < len(data):
            cudi = 0
            repeat = 0
            box = np.zeros(shape=[*self.finalSizeBox()], dtype=self.getNPType())
            while cudi < self.threshCurviDist and postureID < len(data):
                postureNormalizedPlus1 = data[postureID]
                minPosX, minPosY, minPosZ, maxPosX, maxPosY, maxPosZ, center = getMinMaxXYZ(postureNormalizedPlus1)

                postureNormalizedPlus1 = self.normalizePosture(postureNormalizedPlus1,
                                                               np.array([minPosX, minPosY, minPosZ]),
                                                               np.array([maxPosX, maxPosY, maxPosZ]), center,
                                                               self.sizeBox)
                # get the displacement of each joint, then sum them
                positionsPlus1 = [postureNormalizedPlus1.joints[i].position for i in self.jointSelected]
                cudiLocal = sum([np.linalg.norm(np.array(positionsPlus1[i]) - np.array(positions[i])) for i in
                                 range(len(positions))])
                if cudiLocal < self.tresholdToleranceCuDi:
                    postureID += 1
                    repeat += 1
                    continue
                cudi += cudiLocal
                # add the position of each joint if displacement is big enough with the last in the list
                for i in range(len(self.jointSelected)):
                    pos = postureNormalizedPlus1.joints[self.jointSelected[i]].position
                    box[int(pos[2]), int(pos[1]), 0, 0] += repeat + 1


                postureNormalized = postureNormalizedPlus1
                positions = positionsPlus1
                postureID += 1
                repeat += 1
            box[:, :, 0, 0] /= repeat
            repeats.append(repeat + repeatb4)
            repeatb4 = 0

            stdDev = 1.3
            dist = 3
            ck = 1
            for iBone, bones in enumerate(self.morphoLogy.getBones()):
                posJoint1 = postureNormalized.joints[bones[0]].position
                posJoint2 = postureNormalized.joints[bones[1]].position

                bornMinY= int(min(posJoint1[1],posJoint2[1]) - dist)
                bornMaxY = int(max(posJoint1[1],posJoint2[1]) + dist)
                bornMinY = int(max(0, bornMinY))
                bornMaxY = int(min(self.sizeBox[1], bornMaxY))

                bornMinZ= int(min(posJoint1[2],posJoint2[2]) - dist)
                bornMaxZ = int(max(posJoint1[2],posJoint2[2]) + dist)
                bornMinZ = int(max(0, bornMinZ))
                bornMaxZ = int(min(self.sizeBox[0], bornMaxZ))

                for y in range(bornMinY, bornMaxY):
                     for z in range(bornMinZ, bornMaxZ):
                        if y>=self.sizeBox[1] or z>=self.sizeBox[2]:
                            continue
                        boxZY = box[z, y]
                        boxZY[1 + len(self.jointSelected) + iBone, 0] = math.exp(
                                -self.getDistToBoneZY(posJoint1, posJoint2, z, y) ** 2 / (2 * stdDev) * ck)

            for j in range(len(self.jointSelected)):
                pos = postureNormalized.joints[self.jointSelected[j]].position
                distance = 2.5

                posYMin = int(pos[1] - distance)
                posYMax = int(pos[1] + distance)
                posZMin = int(pos[2] - distance)
                posZMax = int(pos[2] + distance)

                for z in range(posZMin+1, posZMax+1):
                    for y in range(posYMin+1, posYMax+1):
                        if z < 0 or z >= self.sizeBox[0] or y < 0 or y >= self.sizeBox[1]:
                            continue
                        box[ z, y, 1 + j, 0] = math.exp(-((z - pos[2]) ** 2 + (y - pos[1]) ** 2) / (2 * stdDev) * ck)

            boxes.append(box)

        # end a chrono
        end = time.time()
        print("time toVoxelize", end - start)

        return boxes, repeats
