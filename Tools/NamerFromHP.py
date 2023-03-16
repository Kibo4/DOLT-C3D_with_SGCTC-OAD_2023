from typing import Dict


def getNameFromHP(hp:Dict)->str:
    dimX, dimY, dimZ = hp["dimensionImage"]
    strDim = str(dimX) + "x" + str(dimY) + "x" + str(dimZ)
    VoxNumber = str(hp["modeVoxelisation"])
    thresoldCudi = "_CuDi" + str(hp["thresholdCuDi"])
    jointsSelectedstr = "_JointsNB" + str(len(hp["jointSelection"]))
    mirrorSeq = hp["mirrorSeq"]
    mirroredSTR = "_mirrored" if mirrorSeq else ""
    splitSize = hp["splitSize"]
    canOutBeforeEnd = hp["doSSG"]

    attribute = "R4D_"+strDim+"_split" + str(splitSize)+thresoldCudi+ jointsSelectedstr + mirroredSTR + "_customRec_Vox"+VoxNumber+"_" + \
            ("SSG" if canOutBeforeEnd else "HSG")
    return attribute