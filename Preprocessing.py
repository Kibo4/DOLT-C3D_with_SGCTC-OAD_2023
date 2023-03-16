"""Preprocessing

The goal of this file is to transform the online signal of each gesture to a 4D tensor:
- temporal (piece of motion, using a fixed length of curvilinear displacement ->$$thesholdCuDi$$ )
- X axis
- Y axis
- heatmaps in channels


Arguments to give:
    - db name (located in the path specified in code)
    - "db.info" file name (in pathDB), this file contains the information about the database, like name, number of classes, etc.
    - "hp.info" file name (in pathDB), this file contains the hyperparameters used (only) for the pre-processing,

To consider folds, add the fold name to the db.info file name, like "dbFold1.info" for fold "Fold1".
it will prepro data to "pathDB/PreprocessedData.....[path constructed with the info of hp.info]_Fold1/"

In pathDB/dbName, the folders should be organized like this:
    - Data: contains the raw data
    - Label: contains the labels files, 1 file per sequence
    - Split: contains the protocol files, the split file is specified in the db.info file
    - Actions.csv: contains the names and id of the actions, 1 per line. The id 0 should be the "nothing" action.
    the format is "id;name" per line

"""

# %%

import os
import sys
import random
from typing import List, Tuple
import numpy as np
from Tools import DataSetReader, CurvilinearDistanceTool
from Tools.DatasetsFormaters.PosturesToFile import PostureToFile
from Tools.FiltererLowPass import FilterThisSequence
from Tools.Gesture.Label import Label
from Tools.Gesture.LabeledSequence import LabeledSequence
from Tools.Gesture.Morphology import Morphology
from Tools.Gesture.MorphologyGetter import MorphologyGetter
from Tools.Gesture.Posture import Posture
import shutil

from Tools.LossesAndMetrics import CustomRecurrenceMatrix
from Tools.LossesAndMetrics.CustomRecurrenceMatrix import exportGraphFromMatrix
from Tools.NamerFromHP import getNameFromHP
from Tools.RepresentationExtractor import MapperIdVoxelizer
import tensorflow as tf
from Tools.RepresentationExtractor.VoxelizerHandler import VoxelizerHandler

# will copy the hyperparams file to the output folder with the generic name "hyperparams_preprocess.info",
# in addition to the name given in argument
hyperParamPreproFileNameDefault = "hyperparams_preprocess.info"

assert len(sys.argv) > 3, "Not enough arguments, need at least 3: dbName, dbInfoFileName, hyperParamPreproFileName"

db = sys.argv[1]
fold = "_" + sys.argv[2].replace("db", "").replace(".info", "")
dbInfoFileName = sys.argv[2]
hyperParamPreproFileName = sys.argv[3]

pathDB, separator = "/srv/tempdd/wmocaer/data/" + db + "/", "/"
# pathDB, separator = "C:\workspace2\\Datasets\\" + db + "/", "/"

# to store the generated graph images
saveGraphImage = False  # to export the graphs generated from the recurrence matrix
pathOutputGraph = pathDB + "GraphImages" + separator

# Raw data path
pathInputData = pathDB + "Data" + separator
pathProtocolFolder = pathDB + "Split" + separator
protocol = pathDB + "Split"

print("Open DB infos", dbInfoFileName)
actionFileName = "Actions.csv"

# Read config file

# database info (device, protocol path, nbClass..)
finfo = open(pathDB + dbInfoFileName, "r")
DBinfos = eval("\n".join(finfo.readlines()))
finfo.close()
pathInputLabel = pathDB + DBinfos["labelFolder"] + separator

# hyperparemters linked to the preprocessing : size of voxelization image, thresholds, subsampling...
finfo = open(pathDB + hyperParamPreproFileName, "r")
hyperparams = eval("\n".join(finfo.readlines()))
finfo.close()

# the association between class id and name
finfo = open(pathDB + actionFileName, "r")
actions = finfo.readlines()
actions = list(map(lambda s: s.split(";")[1].strip(), actions))
finfo.close()

# protocol file (train/test split)
protocolFile = pathProtocolFolder + DBinfos["protocol"]  # get the protocol file, train and test set is specified
finfo = open(protocolFile, "r")
filesTrainTest = finfo.readlines()
finfo.close()

# splitsize is the approximated number of chunks per sequence
# we split the sequence to have  closer sequence length, to have more homogeneous batch during training
# if the sequence is too short, we don't split it
# we do not cut inside a gesture, we wait for the end of the gesture to cut
splitSize = hyperparams["splitSize"]

# refered to the paper
# SSG = Soft Segmentation Guided, it will generate the recurrence matrix for the CTC to do that pruning
# if doSSG is False, then we will produce the HSG pruning
# Note that using this recurrence matrix is not neccecary during training, it wont be used if the hyperparameters for
# the model are not set to use it ("useSegmentationGuidedCTC" hyperparameter)
doSSG = hyperparams["doSSG"]

# there is the possibility to mirror the sequences, to have more data
# note that it is not used in thjis manner in this paper.
# If we set doMirroring here, this will create new sequences, and the labels will be changed accordingly to the
# labelsChangeWhenMirror dictionary.
# This is useful if the symmetry of the gesture is not the same label.
# otherwise the sequence can be just mirror during the training loop, (see "mirrorSeqProba" hyperparameter)
# and the labels will not be changed
mirrorSeq = hyperparams["mirrorSeq"]
doMirroring = True
try:
    labelsChangeWhenMirror = hyperparams["labelsChangeWhenMirror"]
    if labelsChangeWhenMirror is None:
        raise Exception("labelsChangeWhenMirror is None")
except:
    labelsChangeWhenMirror = {}
    doMirroring = False
    assert not mirrorSeq, "You can't mirror the sequence if you don't have the labelsChangeWhenMirror"

# get the name of the preprocessing, to store the data in the right folder
# important to be findable by the training script which will use the same function to get the name
attribute = getNameFromHP(hyperparams) + fold

print("Preprocessing name : ")
print(attribute)
pathOutputTrainPreprocess = pathDB + "PreprocessedData" + attribute + separator
pathOutputValidPreprocess = pathDB + "PreprocessedDataValid" + attribute + separator
pathOutputTestPreprocess = pathDB + "PreprocessedDataTest" + attribute + separator
pathOutputPreprocessLabel = pathDB + "PreprocessedLabel" + attribute + separator

# to export a visual of data in readable format
saveImage = False
pathVoxelizationExport = pathDB + "Voxelized" + attribute + separator

# to export the information of the number of frame per chunk, useful to recover the time information
pathCuDiNbPosturePerSegement = pathDB + "CuDiSplit" + attribute + separator

# to export the filtered data, useful to see the effect of the filtering
saveFilteredGesture = False
pathFilteredExport = pathDB + "Filtered" + attribute + separator

# create the output folder if it does not exist
if not os.path.exists(pathOutputTrainPreprocess):
    os.mkdir(pathOutputTrainPreprocess)
    os.mkdir(pathOutputTestPreprocess)
    os.mkdir(pathOutputValidPreprocess)


def readFileAndAddData(fileData: str, fileLabel: str, actionsNames: List[str], gestures: List[LabeledSequence],
                       fileSampleName):
    """
    add the gestures of the 'file' in the 'gestures' list
    :param fileData: the path of the file containing the data
    :param fileLabel: the path of the file containing the labels
    :param actionsNames: the list of the actions names
    :param gestures: in/out, the list to fill with gestures
    :param fileSampleName: the name of the sample
    :return: void, update the gestures list passed in parameter
    """

    nbSkeleton: int = DBinfos["nbSkeleton"]
    subSampling: int = hyperparams["subSampling"]
    device: str = DBinfos["device"]
    morph: Morphology = MorphologyGetter.getMorphologyFromDeviceName(device)

    seqSkel1: List[Posture]
    seqSkel2: List[Posture]
    seqSkel1, seqSkel2 = DataSetReader.readDataPostures(fileData, nbSkeleton, morph, subSampling)
    assert seqSkel2 == None or len(seqSkel1) == len(seqSkel2)  # same sequence length

    labels: List[Label] = DataSetReader.readLabels(fileLabel, actionsNames, subSampling)
    labeledSeq: LabeledSequence = LabeledSequence(seqSkel1, seqSkel2, labels, fileSampleName)
    gestures.append(labeledSeq)


nbClass = DBinfos["nbClass"]

# idVoxelisation refers to the dictionnary in Tools/RepresentationExtractor/MapperIdVoxelizer.py
idVoxelisation = hyperparams["modeVoxelisation"]

# dimensionImage is the size of the image that will be generated, ie WxHxD, where D is the depth (Z axis)
dimensionEuclideanSpace = np.array(list(hyperparams["dimensionImage"]))

# threshold cudi is the $\theta$ parameter of the paper,
# it is the amount of displacement inside each chunk
thresholdCuDi = hyperparams["thresholdCuDi"]

# not used (set to 0 for the paper), it is to ignore small displacement,
# all displacement under this threshold will be ignored (not counted into the total displacement)
toleranceMoveThreshold = hyperparams["toleranceMoveThreshold"]

# allows to ignore some joints, for example if we want to ignore fingers or other unsignificant joints
# we specify the joints to keep in this parameter
jointsSelected = hyperparams["jointSelection"]

# we use 90% of the "training" data for training, and 10% for validation set
splitForTrain = 90 / 100.

listFilesTrain = list(map(lambda s: s.strip(),
                          filesTrainTest[1].strip().split(",")))  # list of train files is on the 2nd line

"""
Format of the protocol file :
Train files
file9.txt, file2.txt, file3.txt, ...
[Validation files           # Facultative
file8, file45, file12, ...] # Facultative
Test files
file1, file4, file5, ...

"""

# for some dataset, two lines are added to specify the validation set
# in this case, we do not used the 90% computed above.
if len(filesTrainTest) < 6:  # if there is no validation set precised in the protocol file
    random.seed(2)  # to have the same split each time
    random.shuffle(listFilesTrain)
    nbTrain = int(splitForTrain * len(listFilesTrain))
    nbValid = len(listFilesTrain) - nbTrain
    oldListTrain = listFilesTrain
    listFilesTrain = oldListTrain[0:nbTrain]
    listeFilesValid = oldListTrain[nbTrain:]
    listFilesTest = list(
        map(lambda s: s.strip(), filesTrainTest[3].strip().split(",")))  # list of test files is on the 4th line
else:
    listeFilesValid = list(map(lambda s: s.strip(), filesTrainTest[3].strip().split(
        ",")))  # list of train files is on the 2nd line
    listFilesTest = list(
        map(lambda s: s.strip(), filesTrainTest[5].strip().split(",")))  # list of test files is on the 4th line

gesturesTrain: List[LabeledSequence] = []
gesturesValid: List[LabeledSequence] = []
gesturesTest: List[LabeledSequence] = []

# if the dataNames in the split don't have extensions, we can add it here to find the data in the folder Data/
addExtensionStr: str = DBinfos["addExtensionFromProtocol"]
print("--READING DATA--")
print("Training set")
for i, fileSample in enumerate(listFilesTrain):
    if (i == 1 or i % 20 == 0):
        print(i, "/", len(listFilesTrain))
    fileName = separator + fileSample + addExtensionStr
    readFileAndAddData(pathInputData + fileName, pathInputLabel + fileName, actions, gesturesTrain, fileSample)

print("Valid set")
for i, fileSample in enumerate(listeFilesValid):
    if (i == 1 or i % 20 == 0):
        print(i, "/", len(listeFilesValid))
    fileName = separator + fileSample + addExtensionStr
    readFileAndAddData(pathInputData + fileName, pathInputLabel + fileName, actions, gesturesValid, fileSample)

print("Testing set")
for i, fileSample in enumerate(listFilesTest):
    if (i == 1 or i % 20 == 0):
        print(i, "/", len(listFilesTest))
    fileName = separator + fileSample + addExtensionStr
    readFileAndAddData(pathInputData + fileName, pathInputLabel + fileName, actions, gesturesTest, fileSample)

# Filtering data
"""
A small online filter is applied to the data to remove noise.
Butterworth low pass filter is used.
"""

print("--FILTERING DATA--")
def filterLowPass(sequence: LabeledSequence):
    """
    Apply a low pass filter to the data
    Modify IN PLACE the sequence
    :param sequence:
    """
    FilterThisSequence(sequence.postures1)
    if sequence.postures2 is not None:
        FilterThisSequence(sequence.postures2)


for i, gesture in enumerate(gesturesTest):
    if (i == 1 or i % 10 == 0):
        print(i, "on", len(gesturesTest))
    filterLowPass(gesture)

for i, gesture in enumerate(gesturesValid):
    if (i == 1 or i % 10 == 0):
        print(i, "on", len(gesturesValid))
    filterLowPass(gesture)

for i, gesture in enumerate(gesturesTrain):
    if (i == 1 or i % 10 == 0):
        print(i, "on", len(gesturesTrain))
    filterLowPass(gesture)


# Export filtered gestures
def exportFilteredGestures(gesturesSet: List[LabeledSequence]):
    for i in range(len(gesturesSet)):
        if (i == 1 or i % 70 == 0):
            print(i, "on", len(gesturesSet))

        if not os.path.exists(pathFilteredExport):
            os.mkdir(pathFilteredExport)

        gesture = gesturesSet[i]
        if gesture.postures2 is not None:
            PostureToFile.toFile2Sq(gesture.postures1, gesture.postures2, pathFilteredExport + gesture.sequenceName)
        else:
            PostureToFile.toFile(gesture.postures1, pathFilteredExport + gesture.sequenceName)


if saveFilteredGesture:
    exportFilteredGestures(gesturesTest)
    exportFilteredGestures(gesturesValid)
    exportFilteredGestures(gesturesTrain)
    print("done")


# DATA AUGMENTATION : mirroring

def mirrorSequence(sequence: LabeledSequence):
    """
    Mirror the sequence in X axis
    the labels are also modified accordingly to the labelsChangeWhenMirror dictionary
    {A:B} means that if the label is A, it becomes B after mirroring
    It does not mirror if not all labels of the sequence are in the labelsChangeWhenMirror dictionary
    :param sequence:
    :return: mirrored sequence or None if the sequence is not mirrored
    """
    newLabels = []
    labelsClassesid = list(map(lambda l: l.classId, sequence.labels))
    # check if all labels ids are in labelsChangeWhenMirror
    if not all(elem in labelsChangeWhenMirror for elem in labelsClassesid):
        return None

    label: Label
    for label in sequence.labels:
        if label.classId in labelsChangeWhenMirror:
            newLabelClassID = labelsChangeWhenMirror[label.classId]
            newLabel = Label(actions[newLabelClassID], newLabelClassID, label.beginPostureId, label.endPostureId,
                             fileName=label.fileName, actionPoint=label.actionPoint)
            newLabels.append(newLabel)
    posture1 = []
    posture2 = None
    for posture in sequence.postures1:
        posture1.append(posture.mirrorXAxis())

    if sequence.postures2 is not None:
        posture2 = []
        for posture in sequence.postures2:
            posture2.append(posture.mirrorXAxis())

    return LabeledSequence(posture1, posture2, newLabels, sequence.sequenceName)


def doDataMirror(ds):
    """
    Mirror all the sequences in the dataset
    and add them to the dataset
    :param ds: the augmented dataset
    :return:
    """
    seqTrainAugmented = []
    for seq in ds:
        seqAugmented = mirrorSequence(seq)
        if seqAugmented is not None:
            seqTrainAugmented.append(seqAugmented)
    ds += seqTrainAugmented
    return ds


if doMirroring:
    gesturesTrain = doDataMirror(gesturesTrain)
    gesturesValid = doDataMirror(gesturesValid)

# Representation extraction
representationExtractor: VoxelizerHandler

def voxelisationOfThePiecesOfGestures(idVoxelisation:int, gesture: LabeledSequence, thresholdCuDi: float,
                                      toleranceMoveThreshold) -> Tuple[np.ndarray, List[int]]:
    """

    :param idVoxelisation:
    :param gesture: the gesture
    :param thresholdCuDi: the threshold for the cudi
    :param toleranceMoveThreshold: the tolerance for the move threshold
    :return: the images, the number of frames per chunk
    """
    global representationExtractor
    device: str = DBinfos["device"]
    morph: Morphology = MorphologyGetter.getMorphologyFromDeviceName(device)
    voxelized: Tuple[np.ndarray, List[int]]

    representationExtractor = MapperIdVoxelizer.map1sq(idVoxelisation, dimensionEuclideanSpace,
                                                       toleranceMoveThreshold,
                                                       thresholdCuDi,
                                                       jointsSelected, morph)
    voxelizationGesture, numberPosturePerChunk = representationExtractor.extractRepresentation(gesture.postures1)

    return voxelizationGesture, numberPosturePerChunk

# init the representation extractor
voxelizationGesture, numberPosturePerSegment = voxelisationOfThePiecesOfGestures(idVoxelisation, gesturesTest[-1],
                                                                                 thresholdCuDi,
                                                                                 toleranceMoveThreshold)
count = 0
if not os.path.exists(pathCuDiNbPosturePerSegement):
    os.mkdir(pathCuDiNbPosturePerSegement)


def classesInsideTheseBounds(labelsChunkHL: List[Tuple[int, int, int]], startFrameIndex, endFrameIndex) -> List[
    Tuple[int, int, int]]:
    """
    return the classes that are inside the bounds
    :param labelsChunkHL: "High Level" labels in chunk space, with the class id, the start and the end of the class
    :param startFrameIndex: the start of the bounds
    :param endFrameIndex: the end of the bounds
    :return: The labels that are inside the bounds
    """
    listeClass = []
    starts = []
    for classId, start, end in labelsChunkHL:
        if startFrameIndex <= start < endFrameIndex or startFrameIndex < end < endFrameIndex or (
                start <= startFrameIndex <= endFrameIndex <= end):
            realStart = max(0, start - startFrameIndex)  # at least 0, not negative
            if realStart not in starts:
                listeClass.append([classId, realStart, min(end - startFrameIndex, endFrameIndex - startFrameIndex)])
                starts.append(realStart)
    return listeClass


def exportPreprocessData(gestureSet: List[LabeledSequence], numberOfFramePerChunk: List[List[int]], splitInPieces,
                         doSSG):
    """
    can save the images of rpz if saveImages is True
    this function is a generator, it will yield the images, the labels, and recurrent matrix

    :param gestureSet: the gesture set to preprocess
    :param numberOfFramePerChunk: in/out, will contain the number of segment per sequence at the end
    :param splitInPieces: will split the sequence in pieces of size "splitSize"
    :param doSSG: impact the graph construction (for recurrent matrix for CTC),
    if true, allow blank before (below) the end of the gesture, otherwise it is HSG
    :return:
    """
    global count
    for i in range(len(gestureSet)):
        name = gestureSet[i].sequenceName
        print("Preprocessing ", name)
        if (i == 1 or i % 40 == 0):
            print(i, "on", len(gestureSet))
            print("len before voxelisation : " + str(len(gestureSet[i].postures1)))
        images, numberPosturePerSegment = voxelisationOfThePiecesOfGestures(idVoxelisation, gestureSet[i],
                                                                            thresholdCuDi, toleranceMoveThreshold)
        labelsSegments, labelsTemporal, labelsChunksHL = CurvilinearDistanceTool.extractLabelPerFrame_ClassAndWindow(
            gestureSet[i], numberPosturePerSegment, retPerSegmLab=True)

        if (i == 1 or i % 40 == 0):
            print("len after : " + str(len(numberPosturePerSegment)))
        numberOfFramePerChunk.append(numberPosturePerSegment)
        index = 0
        if splitInPieces:
            while index + splitSize + splitSize / 2 < len(labelsSegments):
                try:
                    # to not split inside a gesture
                    indexToSplit = np.where(labelsSegments[index + splitSize:, 0] == 0)[0][0]
                    # take the window (0 value) to split
                except IndexError:
                    # print(file)
                    print("No zero", i)
                    break
                indexToSplit = index + splitSize + indexToSplit
                count += 1
                labelsInvolved = classesInsideTheseBounds(labelsChunksHL, index, indexToSplit)
                customGraphs = CustomRecurrenceMatrix.getCustomGraphMatrixFromBounds(labelsInvolved,
                                                                                     indexToSplit - index,
                                                                                     doSSG)

                if saveGraphImage:
                    exportGraphFromMatrix(customGraphs, labelsInvolved,
                                          pathOutputGraph + str(name) + "index indexToSplit " + str(index) + "_" + str(
                                              indexToSplit) + ".png")
                if len(labelsInvolved) == 0:  # do it after the graph construction
                    labelsInvolved = [[-1, -1, -1]]  # should be accorded to the padding used in training
                yield images[index:indexToSplit], labelsSegments[index:indexToSplit, 1:], \
                      labelsSegments[index:indexToSplit,:1], labelsInvolved, customGraphs
                index = indexToSplit
            count += 1

            # for the last part of the sequence
            labelsInvolved: List[Tuple[int, int, int]] = classesInsideTheseBounds(labelsChunksHL, index,
                                                                                  len(labelsSegments) - 1)

            indexToStart = index
            imagesToKeep = images[indexToStart:]
            customGraphs = CustomRecurrenceMatrix.getCustomGraphMatrixFromBounds(labelsInvolved, len(imagesToKeep),
                                                                                 doSSG)
            if len(labelsInvolved) == 0:  # do it after the graph construction
                labelsInvolved = [[-1, -1, -1]]  # should be accorded to the padding used in training

            if saveGraphImage:
                exportGraphFromMatrix(customGraphs, labelsInvolved, pathOutputGraph + str(name) + ".png")
            # note that labelsSegments[:, :1] is the number of chunks since the gesture has started.
            # it is not used at all in our method anymore.
            yield imagesToKeep, labelsSegments[indexToStart:, 1:],\
                  labelsSegments[indexToStart:,:1], labelsInvolved, customGraphs
        else:
            count += 1
            # note that labelsSegments[:, :1] is the number of chunks since the gesture has started.
            # it is not used at all in our method anymore.
            yield images, labelsSegments[:, 1:], labelsSegments[:, :1]

        f = open(pathCuDiNbPosturePerSegement + name, "w+")
        f.write(",".join(map(str, numberPosturePerSegment)) + "\n")
        f.close()

        if saveImage:
            name = gestureSet[i].sequenceName
            if not os.path.exists(pathVoxelizationExport):
                os.mkdir(pathVoxelizationExport)

            f = open(pathVoxelizationExport + name, "w+")
            f.write(",".join(map(str, images[0].shape)) + "\n")

            for elems in images:
                for x in elems:
                    for y in x:
                        f.write(",".join(map(lambda z: str(float(z)), y)) + "\n")
            f.close()


# %%

def repeatGT(input1, input2, input3, input4, input5):
    """
    Reodering repeating the input2.
    Could be optimized since it is reordered in the Training.
    :param input1: input, dim [seq, dimensionsImage[0], dimensionsImage[1], canal]
    :param input2: output of class, dim [seq,1]
    :param input3: output of window, dim [seq,1]
    :param input4: labelsInvolved,
    :param input5: custom recurrence matrix
    :return:
    """
    return input1, (input2, input2, input3, input4, input5)



def getDataset(generator, isTest):
    dims = representationExtractor.finalSizeBox()
    if isTest:
        output_shapes = (
            tf.TensorShape(
                [None, dims[0], dims[1], dims[2], dims[3]]),  # input image
            tf.TensorShape([None, 1]), tf.TensorShape([None, 1]))
        output_types = (tf.float32, tf.int32, tf.int32)

    else:
        output_shapes = (tf.TensorShape(
            [None, dims[0], dims[1], dims[2], dims[3]]),
                         tf.TensorShape([None, 1]), tf.TensorShape([None, 1]), tf.TensorShape([None, 3]),
                         tf.TensorShape([None, None, None]))
        output_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=output_types,
        output_shapes=output_shapes
    )
    if not isTest:
        dataset = dataset.map(repeatGT, num_parallel_calls=tf.data.AUTOTUNE)  # repeat the GT + one hot encoding
    # if not isValidationSet:
    #     dataset = dataset.shuffle(buffer_size=size, reshuffle_each_iteration=True)
    return dataset


# %%

numberPosturePerSegmentTrain: List[List[int]] = []
numberPosturePerSegmentValid: List[List[int]] = []
numberPosturePerSegmentTest: List[List[int]] = []
datasetTrain = getDataset(
    lambda: exportPreprocessData(gesturesTrain, numberPosturePerSegmentTrain, splitInPieces=True,
                                 doSSG=doSSG), isTest=False)
datasetValid = getDataset(
    lambda: exportPreprocessData(gesturesValid, numberPosturePerSegmentValid, splitInPieces=True,
                                 doSSG=doSSG), isTest=False)
datasetTest = getDataset(
    lambda: exportPreprocessData(gesturesTest, numberPosturePerSegmentTest, splitInPieces=False, doSSG=doSSG), isTest=True)


print("Preprocessing Training set")
count = 0
tf.data.experimental.save(datasetTrain, pathOutputTrainPreprocess)
countTrain = count

print("Preprocessing Validation set")
count = 0
tf.data.experimental.save(datasetValid, pathOutputValidPreprocess)
countValid = count

print("Preprocessing Test set")
count = 0
tf.data.experimental.save(datasetTest, pathOutputTestPreprocess)
countTest = count

print("Preprecessing done. Doing last exports..")

print("Training set size: " + str(countTrain))
print("Validation set size: " + str(countValid))
print("Test set size: " + str(countTest))

f = open(pathOutputTrainPreprocess + "count", "w+")
f.write(str(countTrain))
f.close()

f = open(pathOutputValidPreprocess + "count", "w+")
f.write(str(countValid))
f.close()

f = open(pathOutputTestPreprocess + "count", "w+")
f.write(str(countTest))
f.close()

# %%

# save the parameters dbinfo and hyperparameters
shutil.copyfile(pathDB + dbInfoFileName, pathOutputTrainPreprocess + dbInfoFileName)
shutil.copyfile(pathDB + hyperParamPreproFileName, pathOutputTrainPreprocess + hyperParamPreproFileName)
shutil.copyfile(pathDB + hyperParamPreproFileName, pathOutputTrainPreprocess + hyperParamPreproFileNameDefault)

print("Preprocessing Done")
