from typing import List, Iterable

from Tools.Gesture import Morphology
from Tools.Gesture.Joint import Joint
import numpy as np

from Tools.Gesture.JointType import JointType


class Posture:
    def __init__(self, joints: List[Joint], morphology: Morphology, normalize=False):
        self.joints = joints
        self.morphology = morphology
        if normalize:
            morphology.normalize(self)
        self.mappingJointTypePosition = dict()
        self.updateMapping()

    def updateMapping(self):
        for j in self.joints:
            self.mappingJointTypePosition[j.jointType.id] = j.position

    def toListOfFloat(self) -> List[List[float]]:
        liste = []
        for j in self.joints:
            liste.append(list(j.position))
        return liste

    def shiftAllJoint(self, vector: np.ndarray):
        for j in self.joints:
            j.position = tuple(np.array(j.position) + vector)

    def mirrorXAxis(self):
        from Tools.Gesture.MorphologyGetter import MorphologyGetter
        newJoints = []
        for idJ, j in enumerate(self.joints):
            idJointMirror = MorphologyGetter.getMirrorMember(self.morphology.deviceName, idJ)
            newJoint = Joint(
                self.mirrorXPosition(j.position),
                JointType(idJointMirror))
            newJoints.append(newJoint)
        newJoints = sorted(newJoints, key=lambda j: j.jointType.id)
        return Posture(newJoints, self.morphology)

    def mirrorXPosition(self, pos: tuple) -> np.ndarray:
        return np.array([-pos[0], pos[1], pos[2]])

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Obj.Posture"
