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

    sys.path.extend(['~/workspace/OLT-C3D_OAD'])
    sys.path.append("Tools")
    sys.path.append("Model")
    sys.path.append("..")

import os
import tensorflow as tf
from Evaluation.ExportBounds import exportBoundsOneModel


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "No GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run():
    pathDB = "C:\\workspace2\\Datasets\\Chalearn\\"
    db = "Chalearn"
    pathDB, separator = "/srv/tempdd/wmocaer/data/" + db + "/", "/"
    # separator = "\\"
    PROJECTNAMEWANDB = "OLT-C3D_OAD_focus_on_earliness"
    doAddFold_i_AtEndOfAttributePreproData = False
    def doExportBounds(modelNames, pathOutput, nameOutputFolder):
        print("group", nameOutputFolder)
        for iFold, modelName in enumerate(modelNames):
            exportBoundsOneModel(modelName, pathOutput+modelName+separator, PROJECTNAMEWANDB, pathDB+"PreprocessedDataTest",
                                 pathDB+"CuDiSplit", pathDB+"Split/",
                                 "Fold"+str(iFold) if doAddFold_i_AtEndOfAttributePreproData else "")


    pathFileWithAllGroups = pathDB+"modelRunsGrouped/"
    pathOutputGeneral = pathDB+"expOut/"
    for file in os.listdir(pathFileWithAllGroups):
        f = open(pathFileWithAllGroups + file, "r")
        models = f.readlines()[0].split(",")
        f.close()

        modelNames = models
        nameFolderOut = file
        pathOutput = pathOutputGeneral + nameFolderOut + separator
        if not os.path.exists(pathOutput):
            os.makedirs(pathOutput)


        doExportBounds(modelNames=modelNames, pathOutput=pathOutput, nameOutputFolder=nameFolderOut)
        print("SUCCESS")
if __name__ == "__main__":
    run()