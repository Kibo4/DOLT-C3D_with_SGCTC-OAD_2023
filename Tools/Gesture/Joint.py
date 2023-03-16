from typing import Any, Tuple, Union

import numpy as np

from Tools.Gesture import JointType


class Joint:
    def __init__(self, position : Union[Tuple[float,float,float],np.ndarray], jointType: JointType):
        self.position = position
        self.jointType = jointType


