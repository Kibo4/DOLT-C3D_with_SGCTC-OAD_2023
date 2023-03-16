from typing import Tuple, List

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv3D, MaxPool3D, Add


class OLTC3D_simpleStream(tf.keras.Model):
    """
     Respresent the OLTC3D model with a double stream.
     This model is described in the paper.
     during training and inference, it expects the following input shape:
     (batch_size, time, width, height, channels)
     Note that no temporal pooling is applied, the temporal dimension should is not reduced, and traditional padding
     would break the causal property of the model.
     """

    def __init__(self, nbClass, dropoutVal: float = 0.2, boxSize: Tuple[int, int, int] = (16, 16, 16),
                 denseNeurones: int = 30, denseDropout: float = 0.3, nbFeatureMap: int = 16,
                 dilatationsRates: List[int] = None, numberOfBlock: int = 4, kernelSize=(2, 4, 4),
                 doMaxPoolSpatial: bool = True, poolSize: Tuple[int, int, int] = (1, 4, 4), poolStrides=(1, 4, 4),
                 nbLayerDense=1, doGlobalAveragePooling=True):
        """

        :param nbClass: the number of classes
        :param dropoutVal: dropout of convlutional layers
        :param boxSize: the size of the input box, (width, height,channels), the width is only one view point
        :param denseNeurones: the number of neurons in the dense layer(s)
        :param denseDropout: the dropout of the dense layer(s)
        :param nbFeatureMap: the number of feature maps in the convolutional layers
        :param dilatationsRates: define the content of one block, the number of layers in the block is
                                the length of the list, the values of the list are the dilatation rates
        :param numberOfBlock: the number of blocks in the model, each block is composed of the same layers,
                             number of layers in a block is defined by the len of dilatationRates
        :param kernelSize: the kernel size of the convolutional layers, (time, width, height)
        :param doMaxPoolSpatial: if True, the spatial pooling is applied
        :param poolSize:
        :param poolStrides:
        :param nbLayerDense: number of dense layers after the convolutional layers
        :param doGlobalAveragePooling: if True, the global average pooling is applied at the end of the
                    convolutional layers, allows to remove the witdh and height dimensions
        """
        super(OLTC3D_simpleStream, self).__init__()
        self.numberOfBlock = numberOfBlock
        self.doGlobalAveragePooling = doGlobalAveragePooling

        self.doMaxPoolSpatial = doMaxPoolSpatial
        self.kernelSize1 = kernelSize
        self.nbLayerDense = nbLayerDense
        self.poolSize = None
        self.poolStrides = None
        if self.doMaxPoolSpatial:
            self.poolSize = poolSize
            self.poolStrides = poolStrides
            self.maxPoolSpatialLayerSame = [MaxPool3D(pool_size=self.poolSize, strides=self.poolStrides,
                                                      padding="same") for _ in
                                            range(len(dilatationsRates) * numberOfBlock)]

        self.denseDropout = denseDropout

        self.denseNeurons = denseNeurones
        self.nbFeatureMaps = nbFeatureMap
        self.dropoutValueConv = dropoutVal
        self.boxSize = boxSize
        self.nbClass = nbClass

        if dilatationsRates is None:
            self.dilatationsRates = [1, 2, 4, 8] * numberOfBlock
        else:
            self.dilatationsRates = dilatationsRates * numberOfBlock
        self.theIndexOfMax = self.dilatationsRates.index(max(self.dilatationsRates))
        self.maxVisibility = sum(self.dilatationsRates)

        self.nbFeatureMaps = [nbFeatureMap] * (len(self.dilatationsRates))
        self.nbFeatureMaps = [self.boxSize[-1]] + self.nbFeatureMaps

        # initialise Conv layers
        self.initLayers()

        # Classification layers
        self.layersDense = []
        self.dropoutDense = []
        for i in range(self.nbLayerDense):
            denseLayer = Dense(self.denseNeurons, activation="relu",
                               name="first_layerClassif" + str(i))
            self.layersDense.append(denseLayer)
            dropout1Classif = Dropout(self.denseDropout)
            self.dropoutDense.append(dropout1Classif)

        if self.doGlobalAveragePooling:
            self.globalAveragePooling = tf.keras.layers.TimeDistributed(
                tf.keras.layers.GlobalAveragePooling2D(keepdims=False))

        # output head
        self.classiCTC = Dense(self.nbClass + 1, activation="linear", name="classification_ctc")

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),))
    def log2(self, x):
        return tf.math.log(x) / tf.math.log(2.)

    def initLayers(self):
        self.layersConv = []

        self.dropoutsConv = []
        self.addLayer = Add()

        for id, dilatRate in enumerate(self.dilatationsRates):
            activ = "relu"
            # precise the input shape only for the first layer
            if id == 0:
                conv = Conv3D(filters=self.nbFeatureMaps[id + 1], kernel_size=self.kernelSize1, strides=1,
                              activation=activ,
                              padding="valid",  # the padding will be set manually
                              input_shape=[None, self.boxSize[0], self.boxSize[1], self.boxSize[2]],
                              dilation_rate=[dilatRate, 1, 1], name="ConvLayer" + str(id) + "Dilat" + str(dilatRate))
            else:
                conv = Conv3D(filters=self.nbFeatureMaps[id + 1], kernel_size=self.kernelSize1, strides=1,
                              activation=activ,
                              padding="valid",
                              dilation_rate=[dilatRate, 1, 1], name="ConvLayer" + str(id) + "Dilat" + str(dilatRate))

            self.layersConv.append(conv)

            if self.dropoutValueConv > 0:
                dropOut = Dropout(self.dropoutValueConv)
                self.dropoutsConv.append(dropOut)

    def call(self, x, training=True, **kwargs):
        """

        :param x: shape is (batchSize, nbTimeStep, boxDim[0]*2, boxDim[1], boxDim[2] (features),1 (to squeeze))
        :param training: become automatically true when training, false when testing
        :param kwargs:
        :return: two outputs, the second one is the same as the first one but with softmax applied (used for the second
                loss, per frame and smoothing). The first one is used for the CTC loss (log softmax is applied directly
                                                                                        inside the loss)
                shapes are (batchSize, nbTimeStep, nbClass+1)
        """
        batch = tf.shape(x)[0]
        # the batches are padded to the same length on temporal axis.
        # The masking is done inside the loss functions
        maxlength = tf.shape(x)[1]
        maxlengthX = tf.shape(x)[2]
        maxlengthY = tf.shape(x)[3]

        x = tf.cast(x, dtype=tf.float32)

        x = tf.squeeze(x, axis=-1)
        # [Batch,#segments,Xdim,Ydim,nbChannels]

        # convolutions part, note that all blocks are applies here, there are not separated in the model

        # convolutions part
        for idLayer, layer in enumerate(self.layersConv):
            # causal padding on temporal axis
            # formula to compute the padding for conv is (kernel_size - 1) * dilation_rate
            pads = tf.multiply(layer.dilation_rate, tf.convert_to_tensor(layer.kernel_size) - 1)
            topadSeq, topadX, topadY = pads[0], pads[1], pads[2]

            # the dimension is reduced because of convolutions non padded, fill with zero to left to keep causality
            # for spatial axis, maxpooling is eventually applied and reduce the dimensions (even with padding = same,
            # due to stride!=1)
            # so we pad to compensate, to keep the same size
            topadX = maxlengthX + topadX - tf.shape(x)[2]  # can at be each side
            topadY = maxlengthY + topadY - tf.shape(x)[3]

            # pad on temporal axis in causal way, doesn't matter on spatial axis
            paddings = [[0, 0], [topadSeq, 0], [topadX // 2, topadX - topadX // 2], [topadY // 2, topadY - topadY // 2],
                        [0, 0]]

            x = tf.pad(x, paddings=paddings, mode="CONSTANT",
                       constant_values=0.)  # -> [batch, seq(pad), x(pad),y(pad),channels]

            # Apply the convolution
            x = layer(x)  # x -> [batch, seg(reduced), x(reduced),y(reduced),50]

            # apply dropout
            if self.dropoutValueConv > 0:
                x = self.dropoutsConv[idLayer](x)

            # apply maxpooling
            if self.doMaxPoolSpatial:
                x = self.maxPoolSpatialLayerSame[idLayer](x)
                # shape is [batch, maxlengthSeq, maxlengthX, maxlengthY, nbFeatureMap]

            # Residual connections between blocks
            if idLayer == 0:
                toAddInFuture = x
            elif idLayer % (len(self.dilatationsRates) // self.numberOfBlock - 1) == 0:  # true if we are on the first layer of a block
                x = self.addLayer([x, toAddInFuture])
                toAddInFuture = x

        resConv = x

        if self.doGlobalAveragePooling:
            # global average pooling is applied on the two spatial axis ONLY
            # the dimensions are totally removed since they are average in one value per feature map
            resConv = self.globalAveragePooling(resConv)  # -> [batch, maxlengthSeq,nbFeatureMap]
        else:
            # otherwise we flat the temporal dimension and the spatial dimensions
            # note that in this case, the number of parameters regarding the next dense layer is much higher
            resConv = tf.reshape(resConv,
                                 shape=[batch, maxlength, maxlengthX * maxlengthY[1] * self.nbFeatureMaps[-1]])

        resClassif = resConv

        # Apply dense layers
        for idD, denseLayer in enumerate(self.layersDense):
            resClassif = denseLayer(resClassif)  # [batch, seq(pad), 50]
            resClassif = self.dropoutDense[idD](resClassif)


        # Apply the last dense layer to get the logits
        resForCTC = self.classiCTC(resClassif)  # [batch, sequence, nbClasse+1]

        # Apply the softmax to get the probabilities for the second output
        return resForCTC, tf.nn.softmax(resForCTC)
