"""
Evaluate the time to process the representation extraction
"""

import time
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from Tools import DataSetReader, CurvilinearDistanceTool
from Tools.FiltererLowPass import FilterThisSequence
from Tools.Gesture.Label import Label
from Tools.Gesture.LabeledSequence import LabeledSequence
from Tools.Gesture.Morphology import Morphology
from Tools.Gesture.MorphologyGetter import MorphologyGetter
from Tools.Gesture.Posture import Posture
from Tools.LossesAndMetrics import CustomRecurrenceMatrix
from Tools.RepresentationExtractor import MapperIdVoxelizer

device = "kinectV1"
dimensionEuclideanSpace = [15,15,15]
jointsSelected = [0,2,3,4,7,8,5,9,11,15,17,13,19]
splitSize=200
count=  0
morph: Morphology = MorphologyGetter.getMorphologyFromDeviceName(device)


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

    nbSkeleton = 1
    subSampling = 1

    seqSkel1: List[Posture]
    seqSkel2: List[Posture]
    seqSkel1, seqSkel2 = DataSetReader.readDataPostures(fileData, nbSkeleton, morph, subSampling)
    assert seqSkel2 == None or len(seqSkel1) == len(seqSkel2)  # same sequence length

    labels: List[Label] = DataSetReader.readLabels(fileLabel, actionsNames, subSampling)
    labeledSeq: LabeledSequence = LabeledSequence(seqSkel1, seqSkel2, labels, fileSampleName)
    gestures.append(labeledSeq)


def filterLowPass(sequence: LabeledSequence):
    """
    Apply a low pass filter to the data
    Modify IN PLACE the sequence
    :param sequence:
    """
    FilterThisSequence(sequence.postures1)
    if sequence.postures2 is not None:
        FilterThisSequence(sequence.postures2)


def voxelisationOfThePiecesOfGestures(representationExtractor, gesture: LabeledSequence) -> Tuple[np.ndarray, List[int]]:
    """

    :param idVoxelisation:
    :param gesture: the gesture
    :param thresholdCuDi: the threshold for the cudi
    :param toleranceMoveThreshold: the tolerance for the move threshold
    :return: the images, the number of frames per chunk
    """
    voxelized: Tuple[np.ndarray, List[int]]
    voxelizationGesture, numberPosturePerChunk = representationExtractor.extractRepresentation(gesture.postures1)

    return voxelizationGesture, numberPosturePerChunk

# pathDB = "C:\workspace2\Datasets\Chalearn\\"
pathDB = "/srv/tempdd/wmocaer/data/Chalearn/"

protocolFile = pathDB + "Split\split.txt"  # get the protocol file, train and test set is specified
pathAction = pathDB + "Actions.csv"  # get the protocol file, train and test set is specified
pathInputData = pathDB + "Data\\"
pathInputLabel =  pathDB + "Label\\"
pathOut =  pathDB + "Log\TimeVoxelisation.txt"
finfo = open(protocolFile, "r")
filesTrainTest = finfo.readlines()
listFilesTest = list(
    map(lambda s: s.strip(), filesTrainTest[3].strip().split(
        ",")))  # list of test files is on the 4th linefor i, fileSample in enumerate(listFilesTest):

finfo = open(pathAction, "r")
actions = finfo.readlines()
actions = list(map(lambda s: s.split(";")[1].strip(), actions))
finfo.close()

print("Testing set")
gesturesTest: List[LabeledSequence] = []
print("Reading...")
for i, fileSample in enumerate(listFilesTest):
    if (i == 1 or i % 20 == 0):
        print(i, "/", len(listFilesTest))
    readFileAndAddData(pathInputData + fileSample, pathInputLabel + fileSample, actions, gesturesTest, fileSample)
print("Filtering...")
for i, gesture in enumerate(gesturesTest):
    if (i == 1 or i % 10 == 0):
        print(i, "on", len(gesturesTest))
    filterLowPass(gesture)


representationToEval =  [1,     1,2,3,4,6,7]
thresholdCuDis =        [1E-10, 3,3,3,3,3,3]
toleranceMoveThreshold = 0

strOut = "Representation;Threshold;Time;ResultingFramesAvg;Time/frame\n"

for id,idRpz in enumerate(representationToEval):
    thresholdCuDi = thresholdCuDis[id]
    print("Representation", idRpz, "with threshold", thresholdCuDi)

    representationExtractor = MapperIdVoxelizer.map1sq(idRpz, dimensionEuclideanSpace,
                                                           toleranceMoveThreshold,
                                                           thresholdCuDi,
                                                           jointsSelected, morph)
    # init to avoid initialisation time in the loop
    voxelizationGesture, numberPosturePerSegment = voxelisationOfThePiecesOfGestures(representationExtractor, gesturesTest[-1])

    resulting = 0
    #start chronometer
    start = time.time()
    for gesture in gesturesTest:
        voxelizationGesture, numberPosturePerSegment = voxelisationOfThePiecesOfGestures(representationExtractor, gesture)
        resulting+= len(voxelizationGesture)
    end = time.time()
    print("len gesture set", len(gesturesTest))
    print("Representation", idRpz, "with threshold", thresholdCuDi, "took", end - start, "s", "resulting in",
          resulting/len(gesturesTest), "frames", "avg time is ", (end - start)/len(gesturesTest), "s/frame")
    strOut += str(idRpz) + ";" + str(thresholdCuDi) + ";" + str(end - start) + ";" +\
              str(resulting/len(gesturesTest)) + ";"+ str( (end - start)/len(gesturesTest))+ "s/frame"+"\n"

f = open(pathOut, "w")
f.write(strOut)
f.close()