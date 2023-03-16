from typing import List

from Tools.Gesture.Label import Label
from Tools.Gesture.Posture import Posture


class LabeledSequence:

    def __init__(self, postures1: List[Posture], postures2: List[Posture], labels:List[Label],sequenceName:str):
        """
        :param postures1:
        :param postures2: can be None
        :param labels:
        :param repeat: the list for the association (for CuDi) the number of frames taken for each segment
        :param sequenceName
        """
        self.sequenceName = sequenceName
        self.labels = labels
        self.postures1 = postures1
        self.postures2 = postures2

