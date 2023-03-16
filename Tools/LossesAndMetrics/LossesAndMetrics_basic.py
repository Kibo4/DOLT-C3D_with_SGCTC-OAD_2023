import tensorflow as tf
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy

catCroEntConf = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.AUTO)
catCroEntAux = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.AUTO)
catCroEntSmooth = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.AUTO)
binCroEnt = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.AUTO)
sparseAcc = SparseCategoricalAccuracy()
sparseAcc2 = SparseCategoricalAccuracy()
MSEsmooth = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)

@tf.function
def getWo0TruePredUsingFirstOutput(y_true, y_pred):
    """
    considered accepted when argmax != class 0
    :return:
    - y_trueWo0_accepted : the true labels without the "0" labels and accepted
    - y_predWo0_accepted : the predicted labels without the "0" labels and accepted
    - originalCountWo0 : the number of labels without the "0" labels
    - newCount : the number of labels without the "0" labels and accepted

    """
    # y_true shape = [batch,seq,nbClass] #  because the "0" non-class is not counted
    y_true_accepted_mask = tf.logical_and(y_true[:, :, 0] != 0,  # correspond to non-0 actions,
                                          y_true[:, :, 0] != -1)  # and non-padded
    y_trueWo0 = tf.boolean_mask(y_true[:, :, :1], y_true_accepted_mask, axis=0)  # shape [ batch*seq-masked, ],
    originalCountWo0 = tf.cast(tf.shape(y_trueWo0)[0], tf.float32)

    y_predWo0 = tf.boolean_mask(y_pred, y_true_accepted_mask,
                                axis=0)  # shape [ batch*seq-masked, 1 (rej)+nbClass)]
    # select the accepted predictions
    argMax = tf.argmax(y_predWo0[:, :], axis=1)
    notZeroClassified = tf.not_equal(argMax, 0)


    y_trueWo0 = tf.squeeze(tf.one_hot(tf.cast(y_trueWo0, tf.int32), depth=tf.shape(y_pred)[-1], axis=-1), axis=1)

    y_predWo0_accepted = tf.boolean_mask(y_predWo0, notZeroClassified, axis=0)  # don't take the prediction rejected
    y_trueWo0_accepted = tf.boolean_mask(y_trueWo0, notZeroClassified, axis=0)  # [batch*seg,nbClass]
    newCount = tf.cast(tf.shape(y_trueWo0_accepted)[0], tf.float32)
    return y_trueWo0_accepted, y_predWo0_accepted, originalCountWo0, newCount


@tf.function
def lossPerFrame(y_true, y_pred, weightOfBG):
    """
    Apply a cross entropy loss per frame, with a weight for the background actions
    :param y_true: [batch,seq,1+nbClass]
    :param y_pred: [batch,seq,1+nbClass]
    :param weightOfBG: of weights for frame without actions
    :return:
    """
    y_true_accepted_mask = y_true[:, :, 0] != -1  # correspond to non-0 actions
    y_true = tf.boolean_mask(y_true, y_true_accepted_mask, axis=0)
    y_pred = tf.boolean_mask(y_pred, y_true_accepted_mask, axis=0)

    y_true_accepted_mask = tf.cast(y_true[:, 0] == 0, tf.float32)  # correspond to 0 (non-actions)
    # shape : [batch,seq]
    # [1,1,0,0,0,0,0,1,1,0,0,0,00,0,0]
    weight = y_true_accepted_mask * (weightOfBG - 1) + 1
    # weieightOfBG-1 =-0.5
    # [-0.5,-0.5,0,0,0,0,0-0.5,-0.5,0,0,0]
    # +1
    # [0.5,0.5,1,1,1,1,1,0.5,0.5,1,1,1,1]

    loss = catCroEntAux(y_true, y_pred, weight)
    return loss

@tf.function
def lossSmoothingPredictionCE(y_true, y_pred):
    toLeft = tf.pad(tf.roll(y_pred, shift=-1, axis=1)[:, :-1], paddings=[[0, 0], [0, 1], [0, 0]])
    lossSmooth = catCroEntSmooth(toLeft[:, :, :], y_pred[:, :, :])
    return lossSmooth

@tf.function
def lossSmoothingPredictionMSE(y_true, y_pred):
    toLeft = tf.pad(tf.roll(y_pred, shift=-1, axis=1)[:, :-1], paddings=[[0, 0], [0, 1], [0, 0]])
    lossSmooth = MSEsmooth(toLeft[:, :, :], y_pred[:, :, :])
    return lossSmooth

@tf.function
def TAR_Argmax_allValues_onlyClasses(y_true, y_pred):
    y_true, y_pred, originalCount, newCount = getWo0TruePredUsingFirstOutput(y_true, y_pred)
    return tf.metrics.categorical_accuracy(y_true, y_pred) * newCount / originalCount

@tf.function
def FAR_Argmax_allValues_onlyClasses(y_true, y_pred):
    y_true, y_pred, originalCount, newCount = getWo0TruePredUsingFirstOutput(y_true, y_pred)
    return (1 - tf.metrics.categorical_accuracy(y_true, y_pred)) * newCount / originalCount

@tf.function
def TAR_FAR_Argmax_Diff(y_true, y_pred):
    return TAR_Argmax_allValues_onlyClasses(y_true, y_pred) - FAR_Argmax_allValues_onlyClasses(y_true, y_pred)

@tf.function
def TAR_3FAR_Argmax_Diff(y_true, y_pred):
    return TAR_Argmax_allValues_onlyClasses(y_true, y_pred) - 3 * FAR_Argmax_allValues_onlyClasses(y_true, y_pred)

@tf.function
def RejectRate_Argmax_onlyClasses(y_true, y_pred):
    y_true, y_pred, originalCount, newCount = getWo0TruePredUsingFirstOutput(y_true, y_pred)
    return (originalCount - newCount) / originalCount

@tf.function
def RejectRate_Argmax_allValues(y_true, y_pred):
    # y_true_accepted_mask = tf.not_equal(y_true[:, :,0], 0)  # correspond to non-0 actions
    # y_trueWo0 = tf.boolean_mask(y_true, y_true_accepted_mask, axis=0)[:,:]  # shape [ batch*seq-masked, nbClass+1], remove 0
    batch, seq = tf.shape(y_pred)[0], tf.shape(y_pred)[1]
    originalCountWo0 = tf.cast(batch * seq, tf.float32)
    y_predAccepted = tf.where(tf.reshape(tf.argmax(y_pred[:, :, :], axis=-1), [batch * seq]) == 0)

    newCount = tf.cast(tf.shape(y_predAccepted)[0], tf.float32)
    return (newCount) / originalCountWo0

