from random import randint

import matplotlib.pyplot as plt
from typing import Tuple, List, Iterable
import numpy as np

def getCustomGraphMatrixFromBounds(labels : List[Tuple[int,int,int]], seqLen, doSoftSegGuided):
    """
    the normal recurrence matrix is a matrix of shape [seqLen, 2U+1, 2U+1] where U is the number of labels which is
    the following one for 3 labels :
    blank       [1,1,0,0,0,0,0] => allow to go from blank to blank or blank to label A
    label B     [0,1,1,1,0,0,0] => allow to go from label B to label B, label B to blank or label B to label C
    blank       [0,0,1,1,0,0,0] => allow to go from blank to blank or blank to label A
    label A     [0,0,0,1,1,0,0] => allow from labA to the same labA or labA to blank. Not allowed to go to the next labA
    blank       [0,0,0,0,1,1,0] => allow to go from blank to blank or blank to label A
    label A     [0,0,0,0,0,1,1] => allow from labA to the same labA or labA to blank.
    blank       [0,0,0,0,0,0,1] => allow to go from blank to blank
    This is the same matrix for each frame of the sequence in the normal CTC loss.

    We design here a new recurrence matrix, which is a matrix of shape [seqLen, 2*U+1, 2U+1]
    Note that this time, the matrix is different for each time step,
    as we want to guide the transitions using segmentations

    :param labels: list of labels, [classID, startFrame, endFrame]
    :param seqLen: the len of the sequence (>=labels[-1][2])
    :param doSoftSegGuided: if False, the path must stay in the gesture until the end of it (HSG),
            otherwise if True, it has the possiblity to go directly in the next blank
             after passing in the gesture during one frame. (SSG)
    :return: the recurrence matrix (seqLen, twoUP1, twoUP1)
    """
    twoUP1 = len(labels) * 2 + 1 # 2U+1
    labels.sort(key=lambda x: x[1])
    matrix = np.zeros(shape=[seqLen, twoUP1, twoUP1],dtype=np.int32)
    current = 1
    for id, lab in enumerate(labels):
        classId, start, end = lab
        if start == 0:
            start = 1
        if end >= seqLen - 1:
            end = seqLen - 1
        idWithBlank = id * 2 + 1
        if current < start:
            matrix[current - 1:start, idWithBlank - 1, idWithBlank - 1] = 1
        current = end + 1
        matrix[start - 1:end, idWithBlank - 1, idWithBlank] = 1  # blank to char
        matrix[start - 1:end - 1, idWithBlank - 1, idWithBlank - 1] = 1  # blank to blank
        matrix[start:end, idWithBlank, idWithBlank] = 1  # char to the same char
        if doSoftSegGuided:
            matrix[start:end, idWithBlank, idWithBlank + 1] = 1  # char to blank
            matrix[start + 1:end + 1, idWithBlank + 1, idWithBlank + 1] = 1  # next blank to blank
        matrix[end, idWithBlank, idWithBlank + 1] = 1  # char to blank
        matrix[end - 1, idWithBlank, idWithBlank + 1] = 1  # char to blank
        if id + 1 < len(labels) and start<labels[id + 1][1] <= end + 1: # if the start of the next one is just after the end
            if labels[id + 1][0] != classId:
                matrix[end, idWithBlank, idWithBlank + 2] = 1  # char to next char

    if current < seqLen:
        matrix[current - 1:seqLen , -1, -1] = 1
    return matrix

def exportGraphFromMatrix(matrix,labelInvolved,path):
    import networkx as nx
    l_ext = []
    for lab in labelInvolved:
        l_ext.append(" ")
        l_ext.append(lab[0])
    l_ext.append(" ")
    l_ext = "".join(map(str, l_ext))
    print(l_ext)

    G = nx.DiGraph()

    node_pos = {}
    node_alpha = []
    node_colors = []

    def graphFromMatrix(matrix):
        for t, mat_t in enumerate(matrix):
            for letter in range(len(mat_t)):
                G.add_node((t, letter))
                node_colors.append('black')
                node_alpha.append(0.3)
                node_pos[(t, letter)] = (t, len(l_ext) - letter)
            for idLine, line in enumerate(mat_t):
                for idCol, col in enumerate(line):
                    if mat_t[idLine][idCol] == 1 and t + 1 < len(matrix):
                        G.add_edge((t, idLine), (t + 1, idCol))

    graphFromMatrix(matrix)
    fig = plt.figure(figsize=(40, 10))
    nx.draw_networkx_nodes(G, pos=node_pos, node_color=node_colors, alpha=node_alpha)
    nx.draw_networkx_edges(G, pos=node_pos)
    plt.xlabel('Time step')
    plt.ylabel('Label')
    plt.yticks(range(1, len(l_ext) + 1), labels=reversed(l_ext.replace(' ', '*')))
    plt.xticks(range(0, len(matrix) ), labels=range(0, len(matrix) ))
    fig.savefig(path,dpi=fig.dpi)
    plt.close()
