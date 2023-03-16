from typing import Any

import PIL
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def drawSequence(name, GTStartEnd, predictionBound, predictionBoundWithoutBlank, seqLen, nbClass) -> Any:
    """
    Tool to draw the result of the prediction
    produce an image where
    - the first row is the ground truth
    - the second row is the prediction with blank
    - the third row is the prediction without blank

    Note that a similar tool is present in the evaluation framework.
    This one is used during the training to visualize some intermediate results.
    :param name: str
    :param GTStartEnd: List[ActionID, startFrame, endFrame]
    :param predictionBound: List[ActionID, startFrame, endFrame]
    :param predictionBoundWithoutBlank: List[ActionID, startFrame, endFrame]
    :param seqLen: int, len of the sequence
    :param nbClass: int
    :return: the PIL Image
    """
    cmap = plt.cm.get_cmap("gist_rainbow", nbClass + 1)  # hsv
    # cols = 5
    # figsize = (cols, 20)
    # rows = len(imagesNames) // cols + 1

    step = 1
    nbElem = seqLen
    nbRows = 3 * 3 + 5
    gs = matplotlib.gridspec.GridSpec(nbRows, nbElem // step + 1)
    # ax1 = plt.subplot(gs[:3, :3])
    # ax2 = plt.subplot(gs[0, 3])
    # ax3 = plt.subplot(gs[1, 3])
    # ax4 = plt.subplot(gs[2, 3])
    # ax1.plot(series[0])
    # ax2.plot(series[1])
    # ax3.plot(series[2])
    # ax4.plot(series[3])
    plt.ioff()
    fig = plt.figure(figsize=(nbElem // step + 1, nbRows), dpi=15)
    # plt.tight_layout()
    axs = []
    # plt.suptitle("GT "+str(GTStartEnd[:,0])[1:-1],fontsize=45//3)
    id = 0

    unitWidth = 1 / ((nbElem + 1) / (step))
    # for ax, imName in zip(axs, imagesNames):
    # for i in range(0,seqLen,step):
    #     # imName= imagesNames[i]
    #     # img = mpimg.imread(pathImgForThisSequence+separator+imName)
    #     # ax = fig.add_subplot(rows, cols, i+1)
    #     # ax = plt.subplot(gs[:2,i//step])
    #     # ax.set_title("Class " +str(GT[id])+" f "+str(id),fontsize=7)
    #     # imag = ax.imshow(img)
    #     # accSTR = 'Accepted' if rejection[id] else "Rejected"
    #     # color = 'green' if rejection[id] else "red"
    #     # colorPred = 'green' if prediction[id]==GT[id] else "red"
    #     # plt.yticks([],rotation=90)
    #     plt.xticks([])
    #     # ax.set_ylabel("pred:"+str(prediction[id]), fontsize=10,color=colorPred)
    #     # ax.set_xlabel(accSTR, fontsize=7,color=color)
    #     startDraw = i*unitWidth
    #
    #     axPrediBrut.add_artist(plt.Rectangle((startDraw, 0,), unitWidth, 2, facecolor=cmap(prediction[id])))
    #
    #     id+=step
    axGT = plt.subplot(gs[0:4, :])

    for gt_start_end in GTStartEnd:
        gtId, start, end = gt_start_end
        startSc = ((start) // step) * unitWidth
        rect = plt.Rectangle((startSc, 0), (end // step - start // step + 1) * unitWidth, 2, facecolor=cmap(gtId))
        axGT.add_artist(rect)
        # print("ax,",ax)

    axPred = plt.subplot(gs[4:11, :])

    for gt_start_end in predictionBound:
        gtId, start, end = gt_start_end
        startSc = ((start) // step) * unitWidth
        rect = plt.Rectangle((startSc, 0), ((end) // step - start // step + 1) * unitWidth, 2, facecolor=cmap(gtId))
        axPred.add_artist(rect)

    axPredScnd = plt.subplot(gs[11:14, :])

    for gt_start_end in predictionBoundWithoutBlank:
        gtId, start, end = gt_start_end
        startSc = ((start) // step) * unitWidth
        rect = plt.Rectangle((startSc, 0), ((end) // step - start // step + 1) * unitWidth, 2, facecolor=cmap(gtId))
        axPredScnd.add_artist(rect)

    axs.append(axPred)
    axs.append(axPredScnd)
    axs.append(axGT)
    #
    # axPred = plt.subplot(gs[2,:])
    # axPred.plot([0.5,0.5],"k")
    # axPred.set_xlim(0,1)
    # axPred.set_ylim(0.0,1)

    # print("ax,",ax)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
    plt.subplots_adjust(wspace=0, hspace=0)

    # plt.savefig(str(name)+".svg")
    # plt.savefig(str(name)+".png")
    def fig2img(fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches='tight')
        buf.seek(0)
        img = PIL.Image.open(buf)
        return img

    img = fig2img(fig)
    plt.close(fig)

    return img
