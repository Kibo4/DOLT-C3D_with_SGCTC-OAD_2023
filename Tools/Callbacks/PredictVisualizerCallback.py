from typing import List, Tuple

import numpy as np
import tensorflow as tf
import wandb

from Evaluation import ExportVisualResult
from Tools.Evaluation.BoundedActionTemporalIoUMetric import getId_Start_ActionPoints_End
from Tools.Strategy.StrategyAcceptSimpleArgmax import StrategyAcceptSimpleArgmax


def getCuDiSplit(file: str,pathCuDiSplit) -> List[int]:
    # file = ".".join(file.split(".")[:-1])
    f = open(pathCuDiSplit + file, "r")
    splitCuDi = list(map(int, f.readlines()[0].split(",")))
    f.close()
    return splitCuDi


class PredictVisualizerCallback(tf.keras.callbacks.Callback):

    def __init__(self, datasetTest, frequency, testFiles, pathLabelOriginal, nbClass, pathCuDiSplit):
        """
        :param frequency: if3, then take 1 eleme on 3
        """
        super().__init__()
        self.pathCuDiSplit = pathCuDiSplit
        self.nbClass = nbClass
        self.pathLabelOriginal = pathLabelOriginal
        self.testFiles = testFiles
        self.frequency = frequency
        self.datasetTest = datasetTest
        self.strategy =  StrategyAcceptSimpleArgmax()

    def on_epoch_end(self, epoch,logs=None):
        if epoch % 10 != 0:
            return
        imgs = {"epoch":epoch}
        for i, data in enumerate(list(iter(self.datasetTest))):
            if i % self.frequency != 0:
                continue
            file = self.testFiles[i]
            # if (i % 100 == 0):
            #     print(i, "/", len(testFiles))
            #     print(file)
            input = data[0]
            # GT = tf.squeeze(data[1][0][1]).numpy()
            prediction = self.model(input, False)  # get prediction output

            # windows = prediction[2][0][:,0]
            predictionCTC = prediction[0][0]
            try :
                cuDiSplit = np.array(getCuDiSplit(file,self.pathCuDiSplit))
            except:
                cuDiSplit = np.ones(predictionCTC.shape[0],dtype=np.int32)
                print("no CuDiSplit for ",file)

            boundsPred: List[Tuple[int, int, int]]
            boundsPred, _ = self.strategy.apply(predictionCTC,  cuDiSplit)  # return start, end, class id


            bounds_withoutBlank: List[Tuple[int, int, int]]
            bounds_withoutBlank, _ =  self.strategy.apply(
                tf.pad(predictionCTC[:, 1:], [[0, 0], [1, 0]], constant_values=-9e8),
                 cuDiSplit)  # return start, end, class id

            actionsId_Points = getId_Start_ActionPoints_End(self.pathLabelOriginal, file)
            # classe,start, end
            boundsGT: List[Tuple[int, int, int]] = list(map(
                lambda classe_start_AP_end: (classe_start_AP_end[0], classe_start_AP_end[1], classe_start_AP_end[3]),
                actionsId_Points))
            try :
                fig = ExportVisualResult.drawSequence(file, boundsGT, boundsPred,
                                                      bounds_withoutBlank, np.sum(cuDiSplit), self.nbClass)
                img = wandb.Image(fig, caption="pred_" + str(file))
                imgs[str(file)] = img
            except Exception as e:
                # print the traceback
                import traceback
                traceback.print_exc()
                print(e)
                print("error with file",file," cant draw it")

        wandb.log(imgs,step=epoch)

