from typing import Callable

import numpy as np

from Tools.Gesture import Tree
from Tools.Gesture.JointType import JointType


class Morphology:

    def __init__(self, nbJoints: int, jointDependencies: Tree, jointTypes:[JointType], normalisationFunction: Callable,
                 deviceName: str, dimension: int = 3):
        """
        :param nbJoints:
        :param jointDependencies: the tree of joints dependencies
        :param normalisationFunction: a function to normalize the skeletton at each frame : Posture-> void (in place)
        :param lenWhenFlattenForDilat1: when
        """

        self.jointTypes = jointTypes
        self.deviceName = deviceName
        assert Morphology.verifConsistency(nbJoints, jointDependencies)
        self.dimension = dimension
        self.normalisationFunction = normalisationFunction
        self.jointDependencies:Tree = jointDependencies
        self.nbJoints = nbJoints
        self.nbBones = None
        self.bones = None
        self.bonesMirror = None
        self.mapping = None
        self.mappingUnique = None
        self.adjacencyMatrix = None

    def getJointIdForName(self,name):
        for joint in self.jointTypes:
            if joint.name.lower() == name.lower():
                return joint.id
        return None
    @staticmethod  # TODO
    def verifConsistency(nbJoints, jointDependencies):
        return True

    def getNbBones(self):
        if self.nbBones is not None:
            return self.nbBones
        node = self.jointDependencies.topNode
        self.nbBones = self.getNbBonesRec(node)
        return self.nbBones
    def getBones(self):
        if self.bones is not None:
            return self.bones
        bones = []
        self.getBonesRec(self.jointDependencies.topNode,bones)
        self.bones = bones
        return bones

    def getBonesRec(self, node, bones, mirror=False):
        """

        :param node:
        :return:
        """
        children = node.children
        if mirror:
            children = list(reversed(children))
        for child in children:
            jointTop: JointType = node.value
            jointChild: JointType = child.value
            firstPoint = jointTop.id
            secondPoint = jointChild.id
            bones.append((firstPoint,secondPoint))
        for child in children:
            self.getBonesRec(child, bones,mirror)

    def getBonesMirror(self):
        if self.bonesMirror is not None:
            return self.bonesMirror
        bonesMirror = []
        self.getBonesRec(self.jointDependencies.topNode,bonesMirror, mirror=True)
        self.bonesMirror = bonesMirror
        return bonesMirror

    def getNbBonesRec(self, node):
        countBones = 0
        for child in node.children:
            countBones += 1
            countBones += self.getNbBonesRec(child)
        return countBones

    def getMappingForMirrorBones(self):
        if self.mapping is not None:
            return self.mapping
        mapping = dict()
        bones = self.getBones()
        bonesMirror = self.getBonesMirror()
        for i in range(len(bones)):
            pos = bonesMirror.index(bones[i])
            mapping[i] = pos
        self.mapping = mapping
        return mapping

    def getUniqueMappingForMirrorBones(self):
        if self.mappingUnique is not None:
            return self.mappingUnique
        mapping = dict()
        bones = self.getBones()
        bonesMirror = self.getBonesMirror()
        for i in range(len(bones)):
            if i not in mapping and i not in mapping.values():
                pos = bonesMirror.index(bones[i])
                if i!=pos:
                    mapping[i] = pos
        self.mappingUnique = mapping
        return mapping

    def normalize(self, p):
        """

        :param p: posture
        :return:
        """
        self.normalisationFunction(p)

    def getAdjacencyMatrix(self):
        """
        build the adjacency matrix of the skeleton
        it should be use with MODE_PREPRO = IMAGE2D, otherwise, transformations is needed
        :return:
        """

        if self.adjacencyMatrix is not None:
            return self.adjacencyMatrix

        nbJoints = self.nbJoints
        adjacencyMatrix = np.zeros((nbJoints, nbJoints))
        bones = self.getBones()
        for bone in bones:
            adjacencyMatrix[bone[0]][bone[1]] = 1
        self.adjacencyMatrix = adjacencyMatrix
        return adjacencyMatrix

