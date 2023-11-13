"""
This script is used to train a model on a database.
Arguments to give:
    - db name (located in path specified in code)
    - "db.info" file name (in pathDB), this file contains the information about the database, like name, number of classes, etc.
    - "hp.info" file name (in pathDB), this file contains the hyperparameters used for the pre-processing,
    and not the model hyperparameters which are in a dictionnary in the code.
    - the value for weightPrior model hyperparameter for training

The script will save the weights of the model in the folder "pathDB/Log/modelTimeStamp/Weights/".

To consider folds, add the fold name to the db.info file name, like "dbFold1.info" for fold "Fold1".
it will take the prepro data from "pathDB/PreprocessedData.....[path constructed with the info of hp.info]_Fold1/"
"""
import math
import os
import sys
from datetime import datetime
from shutil import copytree
from typing import List, Tuple, Callable
import tensorflow as tf
import numpy as np
import wandb
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from wandb.integration.keras import WandbCallback

from Evaluation import ExportVisualResult
from Model.OLTC3D_simpleStream import OLTC3D_simpleStream
from Model.OLTC3D_doubleStream import OLTC3D_doubleStream
from Tools import wandbRecuperator
from Tools.Callbacks.PredictVisualizerCallback import PredictVisualizerCallback
from Tools.Evaluation.BoundedActionTemporalIoUMetric import BoundedActionTemporalIoUMetric, getId_Start_ActionPoints_End
from Tools.Gesture.MorphologyGetter import MorphologyGetter
from Tools.LossesAndMetrics.LossesAndMetrics_basic import lossPerFrame, TAR_Argmax_allValues_onlyClasses, TAR_FAR_Argmax_Diff, \
    RejectRate_Argmax_allValues, \
    RejectRate_Argmax_onlyClasses, FAR_Argmax_allValues_onlyClasses,  \
     TAR_3FAR_Argmax_Diff, lossSmoothingPredictionCE
from Tools.LossesAndMetrics.LossesAndMetrics_CTC_CustomRec import lossCTCSimpleCustomRec, \
    lossCTC_Simple_TF, lossCTCSimpleCustomRec_prior, lossCTCLiuNormal
from Tools.NamerFromHP import getNameFromHP
from Tools.Strategy.StrategyAcceptSimpleArgmax import StrategyAcceptSimpleArgmax
from Tools.RepresentationExtractor import MapperIdVoxelizer
from Tools.RepresentationExtractor.VoxelizerHandler import VoxelizerHandler
from Tools.wandbRecuperator import NoFileWeightsFound


PROJETNAME_WANDB= "OLT-C3D_OAD_focus_on_earliness"
ENTITYNAME_WANDB = "intuidoc-gesture-reco"
useModel3D_simpleStream = False
useModel3D_multiStream = True
separator = "/"

# args: db.info, hp.info
assert len(sys.argv) > 4, "not enough arguments, need:" \
                          " db name (located in path specified in code), db.info name (in pathDB), " \
                          "hp.info name (in pathDB), HP weightPrior for training"
db = sys.argv[1]
# pathDB = "/srv/tempdd/wmocaer/data/" + db + "/"
pathDB, separator = "C:\workspace2\\Datasets\\" + db + "/", "/"

dbInfoFileName = sys.argv[2]
hpFile = sys.argv[3]
fold = "_" + dbInfoFileName.replace("db", "").replace(".info", "")  # can be empty
with open(pathDB + hpFile, "r") as f:
    attribute = getNameFromHP(eval(f.read()))
attribute += fold
weightPrior = float(sys.argv[4])

print(dbInfoFileName, "file info,  fold", fold)
print("attribute", attribute)

# construct paths
pathPreprocessedData = pathDB + "PreprocessedData" + attribute + separator
pathPreprocessedValidData = pathDB + "PreprocessedDataValid" + attribute + separator
pathPreprocessedTestData = pathDB + "PreprocessedDataTest" + attribute + separator
pathPreprocessedLabel = pathDB + "PreprocessedLabel" + attribute + separator

# read db.info (which has been copied in the preprocessed folder during the preprocessing)
finfo = open(pathPreprocessedData + dbInfoFileName, "r")
DBinfos = eval("\n".join(finfo.readlines()))
finfo.close()

# path to the folder containing the labels unprocessed,
# which is used for logging the results of validation set during training
pathLabelOriginal = pathDB + DBinfos["labelFolder"] + separator
pathCuDiSplit = pathDB + "CuDiSplit" + attribute + separator

# the preprocessing has been done with the following hyperparameters
# it is stored in the path of the preprocessed data
hyperParamPreproFileName = "hyperparams_preprocess.info"
pathLog = pathDB + "Log" + separator
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available, can run on CPU but it will be (very) slow"

# configAll = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_visible_devices([], 'GPU')


### Read db infos and hyperparameters
# hyperparemters linked to the preprocessing : size of voxelization image, thresholds, subsampling...
finfo = open(pathPreprocessedData + hyperParamPreproFileName, "r")
hyperparams = eval("\n".join(finfo.readlines()))
finfo.close()

idVoxelisation = hyperparams["modeVoxelisation"]
dimensionsImage = np.array(list(hyperparams["dimensionImage"]))
thresholdCuDi = hyperparams["thresholdCuDi"]
toleranceMoveThreshold = hyperparams["toleranceMoveThreshold"]
jointsSelected = hyperparams["jointSelection"]
nbSkeleton = DBinfos["nbSkeleton"]
nbClass = DBinfos["nbClass"]
pathProtocolFolder = pathDB + "Split" + separator

fCount = open(pathPreprocessedData + "count", "r")
countTrain = int(fCount.readlines()[0])
fCount.close()

fCount = open(pathPreprocessedValidData + "count", "r")
countValid = int(fCount.readlines()[0])
fCount.close()

# prepare logging
date = datetime.now().strftime("%Y%m%d-%H%M%S")
pathWeight = pathLog + date + separator + "Weights" + separator

# get the morphology of the device (used for the voxelizer)
morphology = MorphologyGetter.getMorphologyFromDeviceName(DBinfos["device"])
representationExtractor: VoxelizerHandler
representationExtractor = MapperIdVoxelizer.map1sq(idVoxelisation, dimensionsImage, toleranceMoveThreshold,
                                                   thresholdCuDi,
                                                   jointsSelected, morphology)
boxSize = representationExtractor.finalSizeBox()  # size of the image after representation extraction, W * H * (|Joints|+|Bones|+1)

dilatationRates = [1, 2, 4, 8]
config = {
    "batchSize": 4,
    # weight of the CTC loss, weight of CE per frame is (1-weightCTCLoss). if 0, only the per-frame loss is used
    "weightCTCLoss": 1.0,
    "weightPrior": weightPrior,
    "weightSmoothing": 10.0,
    # weight of the background class (no actions), used for per-frame loss only
    "weightOfBG": 1.0,
    "dropoutValueConv": 0.2,
    "denseSize": 70,
    "denseDropout": 0.3,
    "nbFeatureMap": 50,
    "dilatationRates": dilatationRates,
    "numberOfBlocks": 4,
    # probability of mirroring the sequence (for data augmentation)
    "mirrorSeqProba": 0.4,
    "doGlobalAveragePooling": True,
    "doMaxPoolSpatial": True,
    "kernelSize": 3,
    # size of the kernel on the time dimension. Only tested with 2.
    "kernelSizeTemporalMain": 2,
    "poolSize": 3,
    "poolStride": 3,
    "nbDenseLayer": 1,
    "train_size": countTrain,
    "val_size": countValid,
    "boxSize": boxSize,
    "pathPreprocessedData": pathPreprocessedData,
    # use the callback to reduce the learning rate when the loss is not decreasing
    "useReduceLRCallBack": True,
    "useSegmentationGuidedCTC": True,
    # in the case where segmentation guided CTC is not used, and prior is not used
    # the parameter "useCTCLiu" is used to choose between the implementation of
    # the CTC loss of Liu et al. (used to implement the Segmentation-GuidedCTC)
    # or the CTC loss implemented directly in Tensorflow.
    # This avoid to have differences in the results because of the implmentation choice of CTC.
    # However the Tensorflow version is much faster.
    "useCTCLiu": False,
    # label prior will be used in the CTC loss computation (whatever version, either SG CTC or Liu CTC)
    # It can not be used with the Tensorflow implementation of CTC
    "usePrior": True,
    # use the validation set for training (useful for final tuning after that the HP has been fixed)
    # in this case the training will end after "epochMax" epochs
    "useValidationInTrain": False,
    # maximum number of epochs for training, whatever happens
    "epochMax": 600,
    # the path to the folder where the weights will be saved
    "localPathWeight": pathWeight,
    # if resumeRunName is not None, the training will be resumed from the weights of the run with this name
    # it will first try to download weights from wandb, then it will try to load from the local path "localPathWeight"
    "resumeRunName": None,
    # Note that all these parameters will be charged from the previous run if resumeRunName!= None
}
if config["useValidationInTrain"]:
    config["train_size"] = countTrain + countValid
    config["val_size"] = 0

configAll = {**hyperparams, **DBinfos, **config}

tagModel = "3DModel_multiStream"
if useModel3D_simpleStream:
    tagModel = "3DModel"
tags = [configAll["DBname"], tagModel]
resuming = configAll["resumeRunName"] is not None

# configuration of wandb
if resuming:
    # if resuming, we will use the same wandb run as the one we are resuming
    modelName = configAll["resumeRunName"]

    wandBRun = wandbRecuperator.getRun(modelName,projectName=PROJETNAME_WANDB)
    wandBRun = wandb.init(id=wandBRun.id, project=PROJETNAME_WANDB,
                          entity=ENTITYNAME_WANDB, save_code=True, resume="must")
    wandBRun.summary["hasBeenResumed"] = True
    epochToResume = wandb.summary["epoch"]
    wandBRunDir = wandBRun.dir
    configAll["wandbId"] = wandBRun.id
    configAll["wandbName"] = wandBRun.name
    configAll = wandb.config  # will set hyperparameters of resumed run, including pathWeights

    pathModelToLoad: str
    # try to download weights from wandb
    # if it fails, try to load from local path (in config of the previous run)
    try:
        wandbRecuperator.download_weights(modelName, pathLog,projectName=PROJETNAME_WANDB)
        pathModelToLoad = pathLog + modelName + separator + "Weights" + separator
        print("found weights online on wandb, loading them")
    except NoFileWeightsFound as e:
        print(e)
        print("Look for local in " + configAll.localPathWeight + "model")  # get the resumed run path
        if not os.path.exists(configAll.localPathWeight + "model"):
            raise Exception("No weights found, either with wandb and local")
        pathModelToLoad = configAll.localPathWeight
        print("found weights in local folder, loading them")


else:
    wandBRun = wandb.init(project=PROJETNAME_WANDB, entity=ENTITYNAME_WANDB, save_code=True, reinit=True,
                          tags=tags, config=configAll, dir=pathDB + "wandb")
    wandBRunDir = wandBRun.dir
    configAll["wandbId"] = wandBRun.id
    configAll["wandbName"] = wandBRun.name
    modelName = wandBRun.name
    configAll = wandb.config  # will set new hyperparameters when sweep used
config = configAll

# Define the model
if resuming:
    model = tf.keras.models.load_model(pathModelToLoad + "model", compile=False)
else:
    if useModel3D_simpleStream:
        modelToInstanciate = OLTC3D_simpleStream
    elif useModel3D_multiStream:
        modelToInstanciate = OLTC3D_doubleStream
    else:
        raise Exception("No model selected")
    model = modelToInstanciate(nbClass=nbClass, boxSize=config.boxSize,
                               dropoutVal=config.dropoutValueConv, denseNeurones=config.denseSize,
                               denseDropout=config.denseDropout, nbFeatureMap=config.nbFeatureMap,
                               dilatationsRates=config.dilatationRates, numberOfBlock=config.numberOfBlocks,
                               doMaxPoolSpatial=config.doMaxPoolSpatial,
                               poolSize=(1, config.poolSize, config.poolSize),
                               poolStrides=(1, config.poolStride, config.poolStride), nbLayerDense=config.nbDenseLayer,
                               kernelSize=(config.kernelSizeTemporalMain, config.kernelSize, config.kernelSize),
                               doGlobalAveragePooling=config.doGlobalAveragePooling)


# Define the losses
lossSmoothPred: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
lossSmoothPred = lossSmoothingPredictionCE

# %%
perFrameLoss = lambda t, p: (1 - config.weightCTCLoss) * lossPerFrame(t, p, config.weightOfBG) + \
                            config.weightSmoothing * lossSmoothPred(t, p)

if config.useSegmentationGuidedCTC:
    if config.usePrior:
        lossForCTC = lambda y, t: config.weightCTCLoss * lossCTCSimpleCustomRec_prior(y, t,
                                                                                      weightPrior=config.weightPrior)
        print("using segmentation guided CTC with prior of weight " + str(config.weightPrior))
        print("SSG" if config.doSSG else "HSG", "version")
    else:
        lossForCTC = lambda y, t: config.weightCTCLoss * lossCTCSimpleCustomRec(y, t)
        print("using segmentation guided CTC without prior")
        print("SSG" if config.doSSG else "HSG", "version")

else:
    if config.usePrior:
        lossForCTC = lambda t, p: config.weightCTCLoss * lossCTCLiuNormal(t, p, prior=True,
                                                                          weightPrior=config.weightPrior)
        print("using normal CTC with prior of weight " + str(config.weightPrior))
    else:
        if config.useCTCLiu:
            lossForCTC = lambda t, p: config.weightCTCLoss * lossCTCLiuNormal(t, p, prior=False,
                                                                              weightPrior=config.weightPrior)
            print("using normal CTC without prior, Liu implementation")
        else:
            lossForCTC = lambda t, p: config.weightCTCLoss * lossCTC_Simple_TF(t, p, prior=False)
            print("using normal CTC without prior, TF implementation")


# lossSmoothRej = lambda t,p :    lossSmoothingReject(t,p)


# binCrossEntropyLossDecoder = lambda t,p : lossDecoderBinary(t,p)


losses = [[lossForCTC], [perFrameLoss]]


# metricsMainOutput = [TAR_allValues_onlyClasses, FAR_allValues_onlyClasses, RejectRate_allValues_onlyClasses,
#            ClassifyBackgroundOnBackground, RejectRate_allValues, TAR_FAR_Diff,TAR_3FAR_Diff]
#



metricsAux = [TAR_Argmax_allValues_onlyClasses, FAR_Argmax_allValues_onlyClasses,
              RejectRate_Argmax_onlyClasses, TAR_FAR_Argmax_Diff, TAR_3FAR_Argmax_Diff,
              RejectRate_Argmax_allValues]
# metricsCTCCustom = []
metricsCTCCustom = []

opti = tf.keras.optimizers.Adam(learning_rate=0.001)




model.compile(opti,
              loss=losses,
              loss_weights=[1, 1],
              metrics=[metricsCTCCustom, metricsAux])


# %% md

### Prepare input data

# %%


def tofloat32(input1, GT):
    a, b, c, d, recuMatrix = GT
    return tf.cast(input1, tf.float32), (a, b, c, d, recuMatrix)


dimToInvert: dict = representationExtractor.getTranspositionFeaturesForMirroredVoxelization()
# complete with i:i if the key is not present, until finalSizeBox
for i in list(dimToInvert.keys()):
    if dimToInvert[i] not in dimToInvert:
        dimToInvert[dimToInvert[i]] = i
for i in range(0, boxSize[-2]):
    if i not in dimToInvert:
        dimToInvert[i] = i

original_index = tf.constant(list(dimToInvert.keys()))
inverted_index = tf.constant(list(dimToInvert.values()))


def reformulateForCTCLoss_andMirroring(input1, GT):
    a, b, c, d, recuMatrix = GT
    # a : Batch Time 1
    # recuMatrix : None (bach, Time (variable), U(variable), U (variable)
    # d None, U/2 (variable), 3
    mask = a != -1

    shapeD = tf.shape(d)  # None, U/2, 3
    shapeMat = tf.shape(recuMatrix)  # None, Time, U, U
    shapes = tf.concat((shapeD, shapeMat), axis=0)  # 7
    d = tf.reshape(d, [shapes[0] * shapes[1] * shapes[2]])
    recuMatrix = tf.reshape(recuMatrix, [shapes[3] * shapes[4] * shapes[5] * shapes[6]])

    # allTheMatrixIsNotLess1 = tf.reduce_all(tf.reduce_all(mask, axis=-1), axis=-1)
    # countOfValidTimeForEachBatch = tf.reduce_sum(tf.cast(allTheMatrixIsNotLess1,dtype=tf.int32), axis=-1)
    countOfRealTimeForEachBatch = tf.reduce_sum(tf.cast(tf.squeeze(mask, axis=-1), tf.int32), axis=-1)

    allInOneTensor = tf.concat((shapes, d, countOfRealTimeForEachBatch, recuMatrix), axis=0)

    if config.mirrorSeqProba > 0:
        rd = tf.random.uniform((), minval=0, maxval=1)
        if rd < config.mirrorSeqProba:
            # input1 shape : None, Time, X, Y, Features (13+19+1)
            if useModel3D_multiStream:
                # split input1 in 2 parts on the 2 axis
                input1_1, input1_2 = input1[:, :, 0:boxSize[0] // 2, :, :], input1[:, :, boxSize[0] // 2:, :, :]
                # reverse the 2 parts
                input1_1 = tf.reverse(input1_1, axis=[2])
                input1_2 = tf.reverse(input1_2, axis=[2])
                # concat the 2 parts
                input1 = tf.concat((input1_1, input1_2), axis=2)

            else:
                input1 = tf.reverse(input1, axis=[2])

            # get the order of maps considering the preprocessing choice

            # transpose the features maps according to the dictMirrorMember and the order of the maps
            # for each v in dimToInvert, invert input1[:,:,:,:,v] with input1[:,:,:,:,dimToInvert[v]]
            input1 = tf.gather(input1, original_index, axis=-2)  # put it in the right order (regarding the mapping)
            input1 = tf.gather(input1, inverted_index, axis=-2)  # invert it
            # shape of input1 : None, Time, X, Y, Features (13+19+1)

    return tf.cast(input1, tf.float32), (allInOneTensor, a)  # (a, b, c, allInOneTensor,input1)


def configureDataset(dataset, shuffle=False, sizeBufferShuffle=-1, batchOverride=-1, repeat=True):
    if shuffle:
        dataset = dataset.shuffle(buffer_size=sizeBufferShuffle, reshuffle_each_iteration=True)
    # else:
    #     dataset = dataset.map(tofloat32, num_parallel_calls=tf.data.AUTOTUNE)  # repeat the GT + one hot encoding

    toPad = (
        (tf.constant(0, dtype=tf.float32)),
        (tf.constant(-1, dtype=tf.int32),
         tf.constant(-1, dtype=tf.int32),
         tf.constant(-1, dtype=tf.int32),
         tf.constant(-1, dtype=tf.int32),
         tf.constant(-1, dtype=tf.int32)
         )
    )
    # if is4D:
    #     input_shape = tf.TensorShape(
    #         [None, dimensionsImage[0], dimensionsImage[1], dimensionsImage[2],])
    # else:
    input_shape = tf.TensorShape([None, boxSize[0], boxSize[1], boxSize[2], boxSize[3]])

    output_shapes = (input_shape,
                     (tf.TensorShape([None, 1]),
                      tf.TensorShape([None, 1]),
                      tf.TensorShape([None, 1]),
                      tf.TensorShape([None, 3]),
                      tf.TensorShape([None, None, None]),
                      )
                     )
    dataset = dataset.padded_batch(batchOverride if batchOverride != -1 else config.batchSize,
                                   padded_shapes=output_shapes,
                                   padding_values=toPad)
    dataset = dataset.map(reformulateForCTCLoss_andMirroring, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        dataset = dataset.repeat()
    return dataset


datasetTrain = tf.data.Dataset.load(pathPreprocessedData)
datasetValid = tf.data.Dataset.load(pathPreprocessedValidData)

if configAll.useValidationInTrain:  # use validation set in training
    datasetTrain = datasetTrain.concatenate(datasetValid)
    countTrain = countTrain + countValid
    countValid = 0
    datasetValid = None
else:
    datasetValid = configureDataset(datasetValid)

datasetTrain = configureDataset(datasetTrain, True, countTrain // 2)
# datasetValidBatch1NoRepeat = configureDataset(datasetValid, batchOverride=1, repeat=False)

protocolFile = pathProtocolFolder + config["protocol"]  # get the protocol file, train and test set is specified
finfo = open(protocolFile, "r")
filesTrainTest = finfo.readlines()
finfo.close()
indexLineTest = 3 if len(filesTrainTest) < 6 else 5
testFiles = list(
    map(lambda s: s.strip(),
        filesTrainTest[indexLineTest].strip().split(",")))  # list of test files is on the 4nd line
nbTest = len(testFiles)


def configTestDS(dataset):
    # dataset = dataset.map(to3D, num_parallel_calls=tf.data.AUTOTUNE) # repeat the GT + one hot encoding
    dataset = dataset.batch(1)
    return dataset

# will be used only to visualize the bounds predicted by the model, to have an idea
datasetTest = tf.data.Dataset.load(pathPreprocessedTestData)
datasetTest = configTestDS(datasetTest)

### Prepare callbacks for training

# %%


if not os.path.exists(pathLog):
    os.mkdir(pathLog)
pathLogModel = pathLog + date

if not os.path.exists(pathLogModel):
    os.mkdir(pathLogModel)
if not os.path.exists(pathWeight):
    os.mkdir(pathWeight)
# os.mkdir(pathLog+date+separator+"TensorBoard")
monitorToFollow = "val_loss"
modeToUse = "min"
if configAll.useValidationInTrain:
    monitorToFollow = "loss"
    modeToUse = "min"
earlyStop = tf.keras.callbacks.EarlyStopping(monitor=monitorToFollow, verbose=1, patience=70, mode=modeToUse,
                                             min_delta=0.0001)
checkpoint = tf.keras.callbacks.ModelCheckpoint(pathWeight + separator + "model", monitor=monitorToFollow, verbose=1,
                                                save_best_only=True,
                                                # save_weights_only=True,
                                                mode=modeToUse)
predictVisualizerCallback = PredictVisualizerCallback(datasetTest, 8, testFiles, pathLabelOriginal, nbClass,
                                                      pathCuDiSplit)
# tb_callback = tf.keras.callbacks.TensorBoard(log_dir=pathLog+ date+separator+"TensorBoard"+separator+"training",
#                                              profile_batch='10, 20')
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=pathLog+ date+separator+"TensorBoard"+separator+"training", histogram_freq=1)

callbacks = [checkpoint, earlyStop, predictVisualizerCallback, WandbCallback()]
if config.useReduceLRCallBack:
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss' if not configAll.useValidationInTrain else "loss",
                                           factor=0.8,
                                           patience=20, min_lr=0.000001, min_delta=0.0001)
    callbacks.append(reduce_lr_callback)

# %%

fConfig = open(pathLog + date + separator + "config.txt", "w+")
fConfig.write(str(configAll))
fConfig.close()

# %% md

# Fitting

# %%
initialEpoch = 0 if not resuming else epochToResume+1
history = model.fit(datasetTrain, epochs=configAll.epochMax,
                    steps_per_epoch=int(math.ceil(countTrain / config.batchSize)), verbose=2,
                    initial_epoch=initialEpoch,
                    validation_data=datasetValid,
                    validation_steps=int(
                        math.ceil(countValid / config.batchSize)) if datasetValid is not None else None,
                    callbacks=callbacks)  # val_size / batchSize
print("fitted ! ", len(history.history['loss']), " epochs")

# %%------------------------------------------------------------------------------------------------------------
# %%------------------------------------------------------------------------------------------------------------
# %%-------------------------------------------------TESTING part-----------------------------------------------
# %%------------------------------------------------------------------------------------------------------------
# %%------------------------------------------------------------------------------------------------------------
# this is not the final results, it is just to have an idea of the model performance with one metric
# the final results will be computed with the evaluation framework
# (but the BOD Values should be the same)
doTesting = True
if doTesting:
    print("Testing part")
else:
    print("No testing part")


def getCuDiSplit(file: str) -> List[int]:
    f = open(pathCuDiSplit + file, "r")
    splitCuDi = list(map(int, f.readlines()[0].split(",")))
    f.close()
    return splitCuDi


def exportSequencesImagesGTPred(pathLabelOriginal, file, boundsPred, boundsPredWithoutBlank, seqLen, dictOfImage):
    actionsId_Points = getId_Start_ActionPoints_End(pathLabelOriginal, file)

    # classe,start, end
    boundsGT: List[Tuple[int, int, int]] = list(
        map(lambda classe_start_AP_end: (classe_start_AP_end[0], classe_start_AP_end[1], classe_start_AP_end[3]),
            actionsId_Points))
    # print(file, boundsGT)
    fig = ExportVisualResult.drawSequence(file, boundsGT, boundsPred, boundsPredWithoutBlank, seqLen, nbClass)
    img = wandb.Image(fig, caption="pred_" + str(file))
    dictOfImage[str(file)] = img


def BoundedActionTemporalMetric(resultsLoc: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
                                strategyAcceptation: StrategyAcceptSimpleArgmax, canCorrect, minCouverture,
                                TOLERANCE_DETECTION, exportSeqImg=False):
    TruePositive = np.zeros([nbClass + 1])  # TP # nbClass+1 for the non-class
    FalsePositive = np.zeros([nbClass + 1])  # FP
    nb_action_per_class = np.zeros([nbClass + 1])  # we do not predict something at each temporal frame
    NLToD: List[List[float]] = [[] for _ in range(nbClass + 1)]
    savePerSeq = False
    TrueAcceptAt, FalseAcceptAt = [], []
    stringEvalQualif: str = ("_canCorrect" if canCorrect else "_noCorrection") + "_" + str(int(minCouverture * 100))
    dictOfImage = {}
    for res in resultsLoc:
        file = res[0]
        predictions = res[1]
        # rejections  = res[2]
        # windows  = res[4]
        GT = res[2]
        # predictions = tf.argmax(predictions,axis=1).numpy() #+1 because label 0 is nothing , our model does not output the nothing class
        cuDiSplit = np.array(getCuDiSplit(file))

        bounds: List[Tuple[int, int, int]]
        bounds, _ = strategyAcceptation.apply(predictions,  cuDiSplit)  # return start, end, class id

        bounds_withoutBlank: List[Tuple[int, int, int]]
        bounds_withoutBlank, _ = strategyAcceptation.apply(
            tf.pad(predictions[:, 1:], [[0, 0], [1, 0]], constant_values=-9e8),
             cuDiSplit)  # return start, end, class id

        try:
            if exportSeqImg:
                exportSequencesImagesGTPred(pathLabelOriginal, file, bounds, bounds_withoutBlank, np.sum(cuDiSplit),
                                            dictOfImage)
        except Exception as e:
            print("Error drawing sequence", e)
        # print("File ",file)
        TruePositiveFile, FalsePositiveFile, MatConf, nbActionPerClass, \
        Precision_cFile, Recall_cFile, earlinessFile, NLToD_cFile, \
        TPAllFile, FPAllFile, PrecisionFile, RecallFile, NLToDFile, \
        nbActionsTotalGTFile, TrueAcceptAtFile, FalseAcceptAtFile, _ = BoundedActionTemporalIoUMetric(file,
                                                                                                      bounds,
                                                                                                      pathLabelOriginal,
                                                                                                      nbClass,
                                                                                                      canCorrect,
                                                                                                      minCouverture)

        assert sum(nbActionPerClass) == nbActionsTotalGTFile

        for id, classeValues in enumerate(earlinessFile):
            NLToD[id] += classeValues
        TrueAcceptAt += TrueAcceptAtFile
        FalseAcceptAt += FalseAcceptAtFile
        TruePositive += TruePositiveFile
        FalsePositive += FalsePositiveFile
        nb_action_per_class += nbActionPerClass
    wandb.log(dictOfImage)
    # --Per class
    Precision_c = np.divide(TruePositive, TruePositive + FalsePositive, out=np.zeros_like(TruePositive),
                            where=TruePositive + FalsePositive != 0)
    Recall_c = np.divide(TruePositive, nb_action_per_class, out=np.zeros_like(TruePositive),
                         where=nb_action_per_class != 0)
    NLToD_c = np.array([np.average(elem) for elem in NLToD])
    # -- Total, micro average
    TPAll = np.sum(TruePositive)
    FPAll = np.sum(FalsePositive)
    Precision = np.divide(TPAll, TPAll + FPAll, out=np.zeros_like(TPAll), where=TPAll + FPAll != 0)
    Recall = TPAll / (sum(nb_action_per_class))
    flat_earliness = [item for sublist in NLToD for item in sublist]
    NLToD = np.sum(np.array(flat_earliness))
    NLToD = (NLToD / len(np.array(flat_earliness))) if len(np.array(flat_earliness)) != 0 else 0

    FMeasure = ((2 * Recall * Precision) / (Precision + Recall)) if (Precision != 0 or Recall != 0) else 0

    res = "PerClass\n"
    res += "TruePositivePerClass,\n" + str(" ; ".join(list(["{:.2f}".format(v) for v in TruePositive]))) + "\n"
    res += "FalsePositivePerClass,\n" + str(" ; ".join(list(["{:.2f}".format(v) for v in FalsePositive]))) + "\n"
    res += "nbActionPerClass,\n" + str(" ; ".join(list(["{:.2f}".format(v) for v in nb_action_per_class]))) + "\n"
    res += "Precision_PerClass,\n" + str(" ; ".join(list(["{:.2f}".format(v) for v in Precision_c]))) + "\n"
    res += "Recall_PerClass,\n" + str(" ; ".join(list(["{:.2f}".format(v) for v in Recall_c]))) + "\n"
    res += "NLToD_PerClass,\n" + str(" ; ".join(list(["{:.2f}".format(v) for v in NLToD_c]))) + "\n"

    res += "Total\n"
    res += "TruePositives\n" + "{:.2f}".format(TPAll) + "\n"
    res += "FalsePositives\n" + "{:.2f}".format(FPAll) + "\n"
    res += "nbActionsTotal\n" + "{:.2f}".format(sum(nb_action_per_class)) + "\n"
    res += "Precision\n" + "{:.2f}".format(Precision) + "\n"
    res += "Recall\n" + "{:.2f}".format(Recall) + "\n"
    res += "Fmeasure\n" + "{:.2f}".format(FMeasure) + "\n"
    res += "NLToD\n" + "{:.2f}".format(NLToD) + "\n"

    f = open(pathLogModel + separator + "evaluationMetricBound" + stringEvalQualif + ".txt", "w+")
    f.write(res)
    f.close()

    return Precision_c, Recall_c, NLToD_c, Precision, Recall, NLToD, TrueAcceptAt, FalseAcceptAt, sum(
        nb_action_per_class), FMeasure


### add some values in wandb

# %%

import tensorflow.python.keras.backend as K

trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
wandb.log({"TrainableParams": trainable_count})
try:
    def myprint(s):
        print(s)
        with open(pathWeight + separator + "totalWeigth.txt", 'a+') as f:
            f.write(s + "\n")


    model.summary(print_fn=myprint)
except Exception as e:
    print("Problem with weight calculation 2")
    print(e)
###Copie files in wandbDir

copytree(pathLogModel + separator, wandBRunDir + separator + "weights")
if doTesting:
    iterator = list(iter(datasetTest))
    print("len(iterator)", len(iterator))
    # list of the results :
    # Tuples of [testFileName, predictedClass [len=time*nbClass], rejection list (len=time),GroundTruth (len=time)]
    results: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    model = tf.keras.models.load_model(pathWeight + separator + "model", compile=False)
    # model.load_weights(pathWeight+separator+"model")
    for i, data in enumerate(iterator):
        # data shape : ( [batch,time....), ((batch,time,1),) )
        file = testFiles[i]
        if (i % 100 == 0):
            print(i, "/", len(testFiles))
            print(file)
        input = data[0]
        try:
            GT = tf.squeeze(data[1][0][1]).numpy()
        except:
            GT = None
        prediction = model(tf.cast(input, tf.float32), False)  # get prediction output

        # windows = prediction[2][0][:,0]
        predictionCTC = prediction[0][0]
        # predictionClassRej = prediction[0][0]
        # rejection = predictionClassRej[:,0]
        # prediction = predictionClassRej[:,1:]
        # prediction = tf.argmax(prediction,axis=1).numpy()
        # results.append((file,prediction,rejection,predictionCTC,windows,GT))
        results.append((file, predictionCTC, GT))

    strategy = StrategyAcceptSimpleArgmax()
    exptSeqImg = True


    def doEval(TOLERANCE_DETECTION, IoU_MINIMUM, Accept_CORRECTION):
        global exptSeqImg
        Precision_c, Recall_c, NLToD_c, Precision, Recall, NLToD, \
        TrueAcceptAt, FalseAcceptAt, nbActionTotalGT, FMeasure = BoundedActionTemporalMetric(results,
                                                                                             strategy,
                                                                                             Accept_CORRECTION, IoU_MINIMUM,
                                                                                             TOLERANCE_DETECTION,
                                                                                             exptSeqImg)
        exptSeqImg = False
        print("IoU",IoU_MINIMUM, "AcceptCorrection", Accept_CORRECTION, "Precision", Precision, "Recall", Recall,
                "FMeasure", FMeasure, "NLToD", NLToD)


        wandb.summary["BoundedEval_Precision_" + str(IoU_MINIMUM) + "_" + str(Accept_CORRECTION)] = Precision
        wandb.summary["BoundedEval_Recall_" + str(IoU_MINIMUM) + "_" + str(Accept_CORRECTION)] = Recall
        wandb.summary["BoundedEval_FMeasure_" + str(IoU_MINIMUM) + "_" + str(Accept_CORRECTION)] = FMeasure
        wandb.summary["BoundedEval_NTtoD_" + str(IoU_MINIMUM) + "_" + str(Accept_CORRECTION)] = NLToD


    try:

        IoU_toeval = [0, 0.1, 0.25, 0.5, 0.75, 0.95]
        AcceptCorrection_toevl = [True, False]
        for Iou in IoU_toeval:
            for acpt in AcceptCorrection_toevl:
                doEval(TOLERANCE_DETECTION=0, IoU_MINIMUM=Iou, Accept_CORRECTION=acpt)
    except Exception as e:
        print(e)
        print("Problem with evaluation")

wandb.finish()
