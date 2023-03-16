"""
This script is used to export the bounds of the predictions of a model.
The bounds are exported in files in the folder "Bounds" of the model folder.

The bounds are exported in the following format:
    startFrame, endFrame, classId

This script is decomposed :
    1. Specify the paths
        - with the path of the DB and db.info which contains needed information
    2. Specify the model
    3. Specify the protocol
    4. Load the test dataset
    5. Load the model
    6. Predict
    7. Export the bounds in a folder named "Bounds" in the model folder, format per line : classId, startFrame, endFrame
    8. Export the per-frame prediction in a folder named "Frames" in the model folder

Exemple of output
    Bounds
        file15.txt
        file2.txt
 The output The is directly exploitable by the OAD Evaluation framework

"""
if __name__ == "__main__":
    import sys

    sys.path.append("../Tools")
    sys.path.append("../Model")
    sys.path.append("..")

import os
from typing import List, Tuple
import tensorflow as tf
import numpy as np
from Tools import wandbRecuperator, NamerFromHP
from Tools.Strategy.StrategyAcceptSimpleArgmax import StrategyAcceptSimpleArgmax

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available," \
                                  " can run on CPU but it will be slow (just remove the assert if you want to try it)"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def exportBoundsOneModel(modelName, pathModel,  PROJECTNAMEWANDB, pathPreprocessedDataTestWithoutAttribute,
            pathCuDiSplitWithoutAttribute, pathProtocolFolder,attributeFoldSpecific=""):
    """
    example :
        pathDB = "C:\\workspace2\\Datasets\\Chalearn\\"
        separator = "\\"
        # attribute which is to the name of the folder containing the preprocessed data: "PreprocessedDataTest"+attribute
        pathPreprocessedDataTestWithoutAttribute = pathDB + "PreprocessedDataTest"
        pathCuDiSplitWithoutAttribute = pathDB + "CuDiSplit"
        pathProtocolFolder = pathDB + "Split" + separator
    :param pathModel:
    :param PROJECTNAMEWANDB:
    :param pathPreprocessedDataTestWithoutAttribute:
    :param pathCuDiSplitWithoutAttribute:
    :param pathProtocolFolder:
    :param attributeFoldSpecific:
    :return:
    """
    print("\t model", pathModel)

    if (not os.path.exists(pathModel)):
        os.makedirs(pathModel)
        wandbRecuperator.download_weights(modelName, pathModel, projectName=PROJECTNAMEWANDB)
    f = open(pathModel + "Weights/config.txt")
    configParams = f.readlines()
    f.close()
    configParams = eval("\n".join(configParams))

    attribute = NamerFromHP.getNameFromHP(configParams)
    attribute += "_"+attributeFoldSpecific
    # attribute = "R4D_10x10x10_split200_CuDi5_JointsNB13_customRec_Vox4_SSG_"#to remove
    print("\t\t attribute Prepro", attribute)
    pathPreprocessedDataTest = pathPreprocessedDataTestWithoutAttribute + attribute + "/"
    pathCuDiSplit = pathCuDiSplitWithoutAttribute + attribute + "/"

    protocolFile = pathProtocolFolder + configParams[
        "protocol"]  # get the protocol file, train and test set is specified
    # print("protocole,",protocolFile)
    finfo = open(protocolFile, "r")
    filesTrainTest = finfo.readlines()
    finfo.close()
    indexTest = 3 if len(filesTrainTest) < 6 else 5
    testFiles = list(map(lambda s: s.strip(),
                         filesTrainTest[indexTest].strip().split(",")))  # list of test files is on the 4nd line
    nbTest = len(testFiles)

    def configTestDS(dataset):
        dataset = dataset.batch(1)
        return dataset

    # print("pathPreprocessedDataTest",pathPreprocessedDataTest)
    try:
        datasetTest = tf.data.Dataset.load(pathPreprocessedDataTest)
    except Exception as e:
        print("Error while loading the test dataset")
        print("possible cause : the dataset is not generated, run the script PreprocessData.py with the right parameters")
        print("Error :", e)
        exit(-1)

    datasetTest = configTestDS(datasetTest)

    model = tf.keras.models.load_model(pathModel + "Weights/model", compile=False)
    opti = tf.keras.optimizers.Adam()
    model.compile(opti, loss=[], metrics=[])  # just to build

    iterator = iter(datasetTest)

    # list of the results :
    # Tuples of [testFileName, predictedClass [len=time*nbClass], rejection list (len=time),GroundTruth (len=time)]
    # C# syntax to create a tab from 0 to 31
    # int[] tab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    def getCuDiSplit(file: str) -> List[int]:
        f = open(pathCuDiSplit + file, "r")
        splitCuDi = list(map(int, f.readlines()[0].split(",")))
        f.close()
        return splitCuDi

    strategyAcceptation: StrategyAcceptSimpleArgmax = StrategyAcceptSimpleArgmax()
    results: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    print("\t\t", len(testFiles), "files to test, start evaluating")
    for i, data in enumerate(iterator):
        file = testFiles[i]
        # if(i%100==0):
        print("\t\t", i, "/", len(testFiles))
        print(file)
        input = data[0]
        GT = tf.squeeze(data[1][0]).numpy()
        prediction = model(tf.cast(input, tf.float32), False)  # get prediction output

        predictionCTC = prediction[0][0]
        cuDiSplit = np.array(getCuDiSplit(file))

        boundsPrediction, frames = strategyAcceptation.apply(predictionCTC,
                                                             cuDiSplit)  # return start, end, class id
        # prediction = tf.argmax(prediction,axis=1).numpy()
        results.append((file, boundsPrediction, frames))
    if not os.path.exists(pathModel + "Bounds/"):
        os.makedirs(pathModel + "Bounds/" )
    if not os.path.exists(pathModel + "Frames/"):
        os.makedirs(pathModel + "Frames/")
    for file, bounds, perFrame in results:
        strToPrint = "\n".join([", ".join(map(str, b)) for b in bounds])
        f = open(pathModel + "Bounds/" + file, "w+")
        f.write(strToPrint)
        f.close()

        strToPrint = " ".join(map(str, perFrame))
        f = open(pathModel + "Frames/"  + file, "w+")
        f.write(strToPrint)
        f.close()


if __name__ == "__main__":
    # Path of the DB
    pathDB = "C:\\workspace2\\Datasets\\Chalearn\\"
    separator = "\\"
    # attribute which is to the name of the folder containing the preprocessed data: "PreprocessedDataTest"+attribute
    pathPreprocessedDataTestWithoutAttribute = pathDB + "PreprocessedDataTest"
    pathCuDiSplitWithoutAttribute = pathDB + "CuDiSplit"
    pathProtocolFolder = pathDB + "Split" + separator

    # Specify the model
    modelName = "GOODTOTESTLOCAL"
    dbInfoFileName = "db.info"
    pathModel = pathDB + "Log" + separator + modelName + separator
    print("pathModel", pathModel)
    PROJECTNAMEWANDB = "OLT-C3D_OAD_focus_on_earliness"
    exportBoundsOneModel(modelName, pathModel,  PROJECTNAMEWANDB, pathPreprocessedDataTestWithoutAttribute,
                         pathCuDiSplitWithoutAttribute, pathProtocolFolder, "")
