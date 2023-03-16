from typing import List

from Tools.RepresentationExtractor import VoxelizerHandler
from Tools.RepresentationExtractor.VoxelizerRCB_3D_Projection2D_1axisXY_JointID_bones_heatmaps_fuzzy2 import \
    VoxelizerRCB_3D_Projection2D_1axisXY_JointID_bones_heatmaps_fuzzy2
from Tools.RepresentationExtractor.VoxelizerRCB_3D_Projection2D_1axisZY_JointID_bones_heatmaps_fuzzy2 import \
    VoxelizerRCB_3D_Projection2D_1axisZY_JointID_bones_heatmaps_fuzzy2
from Tools.RepresentationExtractor.VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_fuzzy1 import \
    VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_fuzzy1
from Tools.RepresentationExtractor.VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_fuzzy2 import \
    VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_fuzzy2
from Tools.RepresentationExtractor.VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_fuzzy2_trace import \
    VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_fuzzy2_trace
from Tools.RepresentationExtractor.VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_fuzzy3 import \
    VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_fuzzy3
from Tools.RepresentationExtractor.VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_uniform import \
    VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_uniform

dico1sq = {
    1: VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_uniform,
    2: VoxelizerRCB_3D_Projection2D_1axisXY_JointID_bones_heatmaps_fuzzy2,
    3: VoxelizerRCB_3D_Projection2D_1axisZY_JointID_bones_heatmaps_fuzzy2,
    4: VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_fuzzy2,
    5: VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_fuzzy2_trace,
    6: VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_fuzzy1,
    7: VoxelizerRCB_3D_Projection2D_2axis_JointID_bones_heatmaps_fuzzy3,
}


def map1sq(id: int, sizeBoxInit: List[int], thresholdToleranceCuDi: float, threshCurviDist: float,jointSelected: List[int], morphology) -> VoxelizerHandler:
    """
    :param id: the id of the RepresentationExtractor definede in the dico
    :param sizeBoxInit: the initial size of the 3D image (without joint and skeleton ID) (3items)
    :param thresholdToleranceCuDi: tolerance threshold for consideration into CuDi segment
    :param threshCurviDist: the threshold to reach to fill a segment
    :param thresholdToleranceForVoxelization: all displacement between two frame below this threshold wont be drawn
    :param jointSelected: the joints which will be taken into account  in the voxelisation process
    """
    return dico1sq[id](sizeBoxInit, thresholdToleranceCuDi, threshCurviDist, jointSelected, morphology)


def getNbCanalFor(nbSkeleton, idVoxelisation: int, sizeBoxInit: List[int], thresholdToleranceCuDi: float,
                  threshCurviDist: float,jointSelected: List[int], morphology):
    canal = map1sq(idVoxelisation, sizeBoxInit, thresholdToleranceCuDi, threshCurviDist,
                   jointSelected, morphology).finalSizeBox()[2]
    return canal
