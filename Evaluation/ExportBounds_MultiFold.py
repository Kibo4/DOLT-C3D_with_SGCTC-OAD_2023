"""
This script is used to export the bounds of the predictions of a model, considering MULTI-FOLD.
The bounds are exported in files in the folder "Bounds" of the model folder.
Example of output folder content :
    Bounds
        Fold0
            file15.txt
            file2.txt
            ...
        Fold1
            file11.txt
            file22.txt
            ...
        ...
Same structure for the folder "Frames"

The bounds are exported in the following format:
    classId, startFrame, endFrame

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

from Tools import wandbRecuperator
from Tools.Strategy.StrategyAcceptSimpleArgmax import StrategyAcceptSimpleArgmax
from Tools.RepresentationExtractor import MapperIdVoxelizer
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)



pathDB = "C:\\workspace2\\Datasets\\G3D\\"
separator = "\\"


pathOutput = pathDB+"OutputMultiFold_Vox16_wv"+separator # pathModel


modelNames = ["earnest-frost-5923", "ethereal-terrain-5924", "dark-dragon-5925",
                "visionary-glade-5927", "copper-planet-5928", "sleek-waterfall-5929",
                "copper-yogurt-5930", "still-energy-5931", "dandy-serenity-5932", "hearty-sunset-5935"]

for iFold,modelName in enumerate(modelNames):
    print("fold",iFold)
    print("model",modelName)
    # attribute = "R4D_10x10x10_split200_CuDi0.7_Tol0.0_JointsNB13_mirrored_customRec_Vox14_withOut"+"_C4_fold"+str(iFold)
    attribute = "R4D_10x10x10_split200_CuDi0.2_Tol0.0_JointsNB13_mirrored_customRec_Vox16_withOut_"+"Fold"+str(iFold)
    pathPreprocessedDataTest = pathDB + "PreprocessedDataTest" + attribute + separator
    pathCuDiSplit = pathDB + "CuDiSplit" + attribute + separator
    pathProtocolFolder = pathDB + "Split" + separator

    pathModel = pathDB+"Log"+separator+modelName+separator

    if(not os.path.exists(pathModel)):
        wandbRecuperator.download_weights(modelName,pathDB+"Log"+separator)

    #%%

    f = open(pathModel+"Weights/config.txt")
    configParams = f.readlines()
    f.close()
    configParams = eval("\n".join(configParams))

    #%%

    dimensionsImage = np.array(list(configParams["dimensionImage"]))
    nbClass = configParams["nbClass"]

    actionFileName = configParams["actionsIdMap"]
    f = open(pathDB+actionFileName)
    actions  =  f.readlines()
    f.close()
    actions = list(map(lambda s: s.split(";")[1].strip(),actions))

    couverture=0.1
    dilatationRates = configParams["dilatationRates"]
    idVoxelisation = configParams["modeVoxelisation"]
    dimensionsImage = np.array(list(configParams["dimensionImage"]))
    thresholdCuDi = configParams["thresholdCuDi"]
    toleranceMoveThreshold = configParams["toleranceMoveThreshold"]
    thresholdToleranceDrawing = configParams["thresholdToleranceDrawing"]
    jointsSelected = configParams["jointSelection"]
    nbSkeleton = configParams["nbSkeleton"]

    canal = MapperIdVoxelizer.getNbCanalFor(nbSkeleton,idVoxelisation,dimensionsImage,toleranceMoveThreshold,thresholdCuDi,
                                                                        thresholdToleranceDrawing,
                                                                        jointsSelected)

    #%% md

    ### Prepare the test data

    #%%

    protocolFile =  pathProtocolFolder+configParams["protocol"] # get the protocol file, train and test set is specified
    print("protocole,",protocolFile)
    finfo = open(protocolFile, "r")
    filesTrainTest = finfo.readlines()
    finfo.close()
    indexTest = 3 if len(filesTrainTest) < 6 else 5
    testFiles = list(map(lambda s:s.strip(),filesTrainTest[indexTest].strip().split(","))) # list of test files is on the 4nd line
    nbTest = len(testFiles)

    def configTestDS(dataset):
        dataset = dataset.batch(1)
        return dataset
    print("pathPreprocessedDataTest",pathPreprocessedDataTest)
    datasetTest = tf.data.experimental.load(pathPreprocessedDataTest)


    datasetTest = configTestDS(datasetTest)

    model = tf.keras.models.load_model(pathModel+"Weights"+separator +"model",compile=False)
    opti = tf.keras.optimizers.Adam(learning_rate=configParams["learning_rate"])
    model.compile(opti, loss=[], metrics=[])


    iterator = iter(datasetTest)

    #list of the results :
    #Tuples of [testFileName, predictedClass [len=time*nbClass], rejection list (len=time),GroundTruth (len=time)]


    def getCuDiSplit(file:str)->List[int]:
        f = open(pathCuDiSplit+file,"r")
        splitCuDi = list(map(int, f.readlines()[0].split(",")))
        f.close()
        return splitCuDi

    strategyAcceptation: StrategyAcceptSimpleArgmax = StrategyAcceptSimpleArgmax()
    results:List[Tuple[str,np.ndarray,np.ndarray,np.ndarray]] = []
    for i,data in enumerate(iterator):
        file = testFiles[i]
        # if(i%100==0):
        print(i,"/",len(testFiles))
        print(file)
        input = data[0]
        GT = tf.squeeze(data[1][0]).numpy()
        prediction = model(tf.cast(input,tf.float32),False)  # get prediction output

        predictionCTC = prediction[0][0]
        cuDiSplit = np.array(getCuDiSplit(file))

        boundsPrediction, frames = strategyAcceptation.apply(predictionCTC, cuDiSplit)  # return start, end, class id
        # prediction = tf.argmax(prediction,axis=1).numpy()
        results.append((file, boundsPrediction,frames))
    subFoldOut = "fold"+str(iFold)
    if not os.path.exists(pathOutput+"Bounds"+separator+subFoldOut):
        os.makedirs(pathOutput+"Bounds"+separator+subFoldOut)
    if not os.path.exists(pathOutput+"Frames"+separator+subFoldOut):
        os.makedirs(pathOutput+"Frames"+separator+subFoldOut)
    for file,bounds,perFrame in results:
        strToPrint = "\n".join([", ".join(map(str,b)) for b in bounds])
        f = open(pathOutput+"Bounds"+separator+subFoldOut+separator+file,"w+")
        f.write(strToPrint)
        f.close()

        strToPrint = " ".join(map(str,perFrame))
        f = open(pathOutput+"Frames"+separator+subFoldOut+separator+file,"w+")
        f.write(strToPrint)
        f.close()
print("SUCCESS")