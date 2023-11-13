import tensorflow as tf

from Tools.LossesAndMetrics.ctc_normal_customRec_tf import ctc_loss_log_custom, ctc_loss_log_normal, \
    ctc_loss_log_custom_prior, ctc_loss_log_custom_prior_with_recMatrix_computation


def lossCTCSimpleCustomRec(y_true_flat, y_pred):
    """
    :param y_true_flat : [Batch,None] :
                # 7 first values is dimensions :( None(batch), U/2, 3, None(batch), Time, U, U
                # rest: y_true: ([Batch,None(U/2),3] , [Batch],  [Batch,seq, U(padded),U(padded)])
    # :param y_pred: (Batch,Time,1+nbClass)
    :return:
    """
    shapes = y_true_flat[:7] # None, U/2, 3,  None, Time, U, U
    flatted = y_true_flat[7:]
    d_flatted = flatted[:shapes[0]*shapes[1]*shapes[2]]
    rest = flatted[shapes[0]*shapes[1]*shapes[2]:]
    pred_len = rest[:shapes[0]]
    matRec_flatted = rest[shapes[0]:]
    y_true_argmax = tf.reshape(d_flatted,[shapes[0], shapes[1] , shapes[2]])[:,:,0]
    recurrenceMatrix = tf.cast(tf.reshape(matRec_flatted,[shapes[3], shapes[4] , shapes[5], shapes[6]]),tf.float32)



    y_true_argmax_ragged = tf.ragged.boolean_mask(y_true_argmax, y_true_argmax != -1)
    length = y_true_argmax_ragged.row_lengths(
        axis=1)  # <tf.Tensor: shape=(2,), dtype=int64, numpy=array([7, 8], dtype=int64)>

    vectorForCTC = y_pred[:, :, :]

    loss = ctc_loss_log_custom(vectorForCTC,
                            y_true_argmax,
                            pred_len[:, tf.newaxis], length[:, tf.newaxis],recurrenceMatrix)
    return tf.reduce_mean(loss)

def lossCTCSimpleCustomRec_prior(y_true_flat, y_pred, weightPrior):
    """
    :param prior:
    :param y_true_flat : [None] : 7first values is dimensions :( None(batch), U/2, 3, None(batch), Time, U, U
    # :param y_true: ([Batch,None(U/2),3] , [Batch],  [Batch,seq, U(padded),U(padded)])
    # :param y_pred: (Batch,Time,1+nbClass)
    :return:
    """
    shapes = y_true_flat[:7] # None, U/2, 3,  None, Time, U, U
    flatted = y_true_flat[7:]
    d_flatted = flatted[:shapes[0]*shapes[1]*shapes[2]]
    rest = flatted[shapes[0]*shapes[1]*shapes[2]:]
    pred_len = rest[:shapes[0]]
    matRec_flatted = rest[shapes[0]:]
    y_true_argmax = tf.reshape(d_flatted,[shapes[0], shapes[1] , shapes[2]])[:,:,0] # [Batch, U/2]
    recurrenceMatrix = tf.cast(tf.reshape(matRec_flatted,[shapes[3], shapes[4] , shapes[5], shapes[6]]),tf.float32)


    y_true_argmax_ragged = tf.ragged.boolean_mask(y_true_argmax, y_true_argmax != -1)
    length = y_true_argmax_ragged.row_lengths(
        axis=1)  # <tf.Tensor: shape=(2,), dtype=int64, numpy=array([7, 8], dtype=int64)>

    vectorForCTC = y_pred[:, :, :]
    loss = ctc_loss_log_custom_prior(vectorForCTC,
                            y_true_argmax,
                            pred_len[:, tf.newaxis], length[:, tf.newaxis],recurrenceMatrix,weightPrior)

    return tf.reduce_mean(loss)

def lossCTCSimpleCustomRec_prior_cumputationMatrixInLoss(y_true_flat, y_pred, weightPrior,doSSG):
    """
    :param prior:
    :param y_true_flat : [None] : 7first values is dimensions :( None(batch), U/2, 3, None(batch), Time, U, U
    # :param y_true: ([Batch,None(U/2),3] , [Batch],  [Batch,seq, U(padded),U(padded)])
    # :param y_pred: (Batch,Time,1+nbClass)
    :return:
    """
    shapes = y_true_flat[:7] # None, U/2, 3,  None, Time, U, U
    flatted = y_true_flat[7:]
    d_flatted = flatted[:shapes[0]*shapes[1]*shapes[2]]
    rest = flatted[shapes[0]*shapes[1]*shapes[2]:]
    pred_len = rest[:shapes[0]]
    matRec_flatted = rest[shapes[0]:]
    y_true_argmax = tf.reshape(d_flatted,[shapes[0], shapes[1] , shapes[2]])[:,:,0] # [Batch, U/2, 3]
    y_true_argmax_3 = tf.reshape(d_flatted,[shapes[0], shapes[1] , shapes[2]])[:,:,:] # [Batch, U/2, 3]
    # recurrenceMatrix = tf.cast(tf.reshape(matRec_flatted,[shapes[3], shapes[4] , shapes[5], shapes[6]]),tf.float32)


    y_true_argmax_ragged = tf.ragged.boolean_mask(y_true_argmax, y_true_argmax != -1)
    length = y_true_argmax_ragged.row_lengths(
        axis=1)  # <tf.Tensor: shape=(2,), dtype=int64, numpy=array([7, 8], dtype=int64)>

    vectorForCTC = y_pred[:, :, :]
    loss = ctc_loss_log_custom_prior_with_recMatrix_computation(vectorForCTC,
                            y_true_argmax_3,
                            pred_len[:, tf.newaxis], length[:, tf.newaxis],weightPrior,doSSG)

    return tf.reduce_mean(loss)

def lossCTCLiuNormal(y_true_flat,y_pred,prior=False,weightPrior=1.0):
    """
    :param y_true_flat : [Batch,None] : 7first values is dimensions :( None(batch), U/2, 3, None(batch), Time, U, U
    # :param y_true: ([Batch,None(U/2),3] , [Batch],  [Batch,seq, U(padded),U(padded)])
    # :param y_pred: (Batch,Time,1+nbClass)
    :return:
    """
    shapes = y_true_flat[:7] # None, U/2, 3,  None, Time, U, U
    flatted = y_true_flat[7:]
    d_flatted = flatted[:shapes[0]*shapes[1]*shapes[2]]
    rest = flatted[shapes[0]*shapes[1]*shapes[2]:]
    pred_len = rest[:shapes[0]]
    # matRec_flatted = rest[shapes[0]:]
    y_true_argmax = tf.reshape(d_flatted,[shapes[0], shapes[1] , shapes[2]])[:,:,0]
    # recurrenceMatrix = tf.cast(tf.reshape(matRec_flatted,[shapes[3], shapes[4] , shapes[5], shapes[6]]),tf.float32)

    y_true_argmax_ragged = tf.ragged.boolean_mask(y_true_argmax, y_true_argmax != -1)
    length = y_true_argmax_ragged.row_lengths(
        axis=1)  # <tf.Tensor: shape=(2,), dtype=int64, numpy=array([7, 8], dtype=int64)>

    vectorForCTC = y_pred[:, :, :]

    if prior:
        loss = ctc_loss_log_normal(vectorForCTC,
                            y_true_argmax,
                            pred_len[:, tf.newaxis], length[:, tf.newaxis],prior=True,weightPrior=weightPrior)
    else:
        loss = ctc_loss_log_normal(vectorForCTC,
                            y_true_argmax,
                            pred_len[:, tf.newaxis], length[:, tf.newaxis],prior=False,weightPrior=0)
    return tf.reduce_mean(loss)

def safe_log(x, eps=1e-5):
  """Avoid taking the log of a non-positive number."""
  safe_x = tf.where(x <= 0.0, eps, x)
  return tf.math.log(safe_x)

def lossCTC_Simple_TF(y_true_flat,y_pred,prior = False):
    y_true_argmax, length, pred_len = getTensorsFromFlattedY_true(y_true_flat)
    if False :
        sm = y_pred
        avg_sm = tf.stop_gradient(tf.reduce_mean(sm, axis=0, keepdims=True))  # (1,1,dim)
        y_pred = tf.math.divide_no_nan(y_pred, avg_sm)
        # y_pred = tf.exp(y_pred)

    loss = tf.nn.ctc_loss(labels=y_true_argmax, logits=y_pred,  # should be logits (before softmax)
                          label_length=length,
                          logit_length=pred_len,
                          logits_time_major=False, blank_index=0)
    return tf.reduce_mean(loss)

def getTensorsFromFlattedY_true(y_true_flat):
    shapes = y_true_flat[:7]  # None, U/2, 3,  None, Time, U, U
    flatted = y_true_flat[7:]
    d_flatted = flatted[:shapes[0] * shapes[1] * shapes[2]]
    rest = flatted[shapes[0] * shapes[1] * shapes[2]:]
    pred_len = rest[:shapes[0]]
    # matRec_flatted = rest[shapes[0]:]
    y_true_argmax = tf.reshape(d_flatted, [shapes[0], shapes[1], shapes[2]])[:,:,0]
    # recurrenceMatrix = tf.cast(tf.reshape(matRec_flatted, [shapes[3], shapes[4], shapes[5], shapes[6]]), tf.float32)

    y_true_argmax_ragged = tf.ragged.boolean_mask(y_true_argmax, y_true_argmax != -1)
    length = y_true_argmax_ragged.row_lengths(
        axis=1)  # <tf.Tensor: shape=(2,), dtype=int64, numpy=array([7, 8], dtype=int64)>
    return y_true_argmax,length,pred_len

