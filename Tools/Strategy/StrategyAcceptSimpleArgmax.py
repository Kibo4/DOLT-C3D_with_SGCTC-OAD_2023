from typing import List

import numpy as np
import tensorflow as tf


class StrategyAcceptSimpleArgmax:
    def __init__(self):
        super().__init__()

    def apply(self, prediction: List[int], repeat: List[int]) -> object:
        """
        returns bounds considering online stragy, no past frames are requalified after seeing future frames
        :param prediction: 1+nbClass, index 0 is for reject. List in CuDi space
        :param repeat: temporal mapping,
        :return:    -list[classiD,debut,fin,], in temporal space,
                    -the associated per frame prediction in temporal space, length = sum(repeat) (temporal length)
        """
        assert len(prediction) != 0
        if len(prediction) != len(repeat):
            print("len(prediction) != len(repeat)")
            print(len(prediction), len(repeat))
        assert len(prediction) == len(repeat)
        prediction = tf.argmax(prediction, axis=1).numpy() - 1  # to have -1 for reject/blank, just for clarty

        listeFinalePred = np.zeros([sum(repeat)], dtype=int) - 1 # temporal domain, set -1 everywhere
        cumulativeSumRepeat = np.cumsum(repeat) - 1  # -1 to correspond with temporal index
        """
        Example:
        repeat = [2,3,4,3]
        means that for the first chunk, two frames has been used, for the second chunk, 3 frames has been used..
        
        the cumulative less 1 is [1,4,8,11]
        for the first chunk, frames of index 0,1 (0-based) are used
        for the second chunk, frames of index 2,3,4 (0-based) are used
        if we predict the class A for the second chunk, with the online strategy, we can say that 
        We consider taht the action A started from frame 4 (0-based), 
        (cant requalify frame 2 and 3 because frame 4 has been used to make the prediction)
        and that it continues until the end of the next chunk (frame 8)
        """
        assert len(repeat) == len(prediction)
        # map in temporal domain, respecting online contraints explained just over
        for id in range(1, len(cumulativeSumRepeat)):
            # get the prediction of the chunk
            prediM1 = prediction[id - 1]
            # stay from the end of last frame, to the next
            listeFinalePred[cumulativeSumRepeat[id - 1]:cumulativeSumRepeat[id]] = prediM1

        currClass = listeFinalePred[0]
        startIndexCurrentClass = 0

        bounds = []
        # compacting frames with the same class
        for i in range(1, len(listeFinalePred)):
            pred = listeFinalePred[i]
            if pred != currClass or pred == -1 or i == len(listeFinalePred) - 1:
                if currClass != -1:
                    if i == len(listeFinalePred) - 1:
                        endBound = i
                    else:
                        # to avoid two classes in the same frame
                        endBound = i - 1
                    bounds.append(
                        [currClass + 1, startIndexCurrentClass,
                         endBound])  # +1 to have real classes indexes (we did -1 at the beginning)
                if i == len(listeFinalePred) - 1 and pred != currClass and pred != -1:
                    bounds.append([pred + 1, i, i])
                currClass = pred
                startIndexCurrentClass = i
        return bounds, listeFinalePred + 1
