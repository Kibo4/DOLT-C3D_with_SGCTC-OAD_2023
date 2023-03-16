import tensorflow as tf
"""
These implementations are based on the implementation of CTC of this paper:
Connectionist Temporal Classification with Maximum Entropy Regularization
Hu Liu, Sheng Jin and Changshui Zhang. Neural Information Processing Systems (NeurIPS), 2018.
https://github.com/liuhu-bigeye/enctc.crnn


"""

@tf.function
def m_eye(n, k=0):
    """

    :param n: size of the matrix
    :param k: the shift value
    :return: a identity matrix where the ones aren shifted (to right) by k
     n=10 , k = 1,
tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    return tf.cast(tf.concat((tf.concat(
                                (tf.zeros([n-k, k]),  tf.eye(n-k)), axis=1),
                         tf.zeros([k, n])), axis=0),tf.float32)
@tf.function
def log_batch_dot(alpha_t, rec):
    '''
    alpha_t: (batch, 2U+1)
    rec: (batch, 2U+1, 2U+1)
     The function performs a mathematical operation called batch dot product, which involves adding
     the values of two matrices and then taking the exponential of the sum.
      The function returns a tensor of the same shape as alpha_t.
    '''
    eps_nan = -1e8
    # a+b
    _sum = tf.repeat(alpha_t[:, :, None], tf.shape(alpha_t)[-1], axis=-1) + rec  # batch;2U+1,2U+1
    _max_sum = tf.reduce_max(_sum, axis=1)
    nz_mask1 = _max_sum > eps_nan  # max > eps_nan
    nz_mask2 = _sum > eps_nan  # item > eps_nan

    # a+b-max
    _sum = _sum - _max_sum[:, None]

    # exp
    _exp = tf.zeros_like(_sum, tf.float32)  # .type(floatX)
    _exp = tf.where(nz_mask2, tf.exp(_sum), _exp)

    # sum exp
    _sum_exp = tf.reduce_sum(_exp, axis=1)

    out = tf.ones_like(_max_sum, tf.float32) * eps_nan
    out = tf.where(nz_mask1, tf.math.log(_sum_exp) + _max_sum, out)
    return out

@tf.function
def log_sum_exp_axis(a, uniform_mask=None, dim=0):
    assert dim == 0
    eps_nan = -1e8
    eps = 1e-26
    _max = tf.reduce_max(a, axis=dim)

    if not uniform_mask is None:
        nz_mask2 = (a > eps_nan) * uniform_mask
        nz_mask1 = (_max > eps_nan) * (tf.reduce_max(uniform_mask, axis=dim) >= 1.)
    else:
        nz_mask2 = a > eps_nan
        nz_mask1 = _max > eps_nan
    # print("nz_mask2.shape",nz_mask2.shape)
    # print("nz_mask1.shape",nz_mask1.shape)

    # a-max
    a = a - _max[None]

    # exp
    _exp_a = tf.zeros_like(a, tf.float32)
    _exp_a = tf.where(nz_mask2, tf.exp(a), _exp_a)

    # sum exp
    _sum_exp_a = tf.reduce_sum(_exp_a, axis=dim)

    out = tf.ones_like(_max, tf.float32) * eps_nan
    out = tf.where(nz_mask1, tf.math.log(_sum_exp_a + eps) + _max, out)
    return out

@tf.function
def log_sum_exp(*arrs):
    #    return T.max(a.clone(), b.clone()) + T.log1p(T.exp(-T.abs(a.clone()-b.clone())))
    c = tf.concat(list(map(lambda x: x[None], arrs)), axis=0)
    # print("c.shape",c.shape)
    return log_sum_exp_axis(c, dim=0)

@tf.function
def ctc_loss_log_custom(pred, token, pred_len, token_len, recurrenceMatrix):
    '''
    Segmentation Guided (SG) CTC loss
    it uses the recurrence matrix to guide the alignment during the training
    the recurrence matrix is a matrix of size (2U+1, 2U+1) where U is the maximum length of the token, it is produced
    during the preprocessing of the data
    :param pred: (batch,Time, voca_size+1)
    :param token: (batch, U)
    :param pred_len: (batch,1)
    :param token_len: (batch,1)
    :param recurrenceMatrix : (batch, 2U+1, 2U+1). More information about its construction is given in the file
                                "CustomRecurrenceMatrix.py"
    :return SG loss: (batch,1)
    '''
    pred = tf.transpose(pred, [1, 0, 2])  # -> ( Time,batch, voca_size+1)
    pred = tf.math.log_softmax(pred)
    Time, batch = tf.shape(pred)[0], tf.shape(pred)[1]
    U = tf.shape(token)[1]
    eps_nan = -1e8

    # token_with_blank
    token_with_blank = tf.concat((tf.zeros([batch, U, 1], dtype=tf.int32), tf.cast(token[:, :, tf.newaxis], tf.int32)),
                                 axis=2)  # (batch, U,2)
    token_with_blank = tf.reshape(token_with_blank, [batch, -1])  # (batch, 2*U)
    # add a blank at the end of elems
    token_with_blank = tf.concat((token_with_blank, tf.zeros([batch, 1], dtype=tf.int32)), axis=1)  # (batch, 2U+1)
    # token_with_blank: [blank, index_e1,blank, index_e2, .... eU, blank]

    length = tf.shape(token_with_blank)[1]  # 2U+1

    # construct the CTC graph
    pred = tf.gather(pred, tf.repeat(token_with_blank[tf.newaxis, :, :], Time, axis=0), axis=2,
                     batch_dims=2)  # (T, batch, 2U+1)]

    # recurrence relation
    # masking the padded data
    recurrenceMatrix = tf.where(recurrenceMatrix != -1, recurrenceMatrix, tf.zeros_like(recurrenceMatrix, tf.float32))
    # use a small value to mask the padded data
    recurrence_relation = eps_nan * (tf.ones_like(recurrenceMatrix, tf.float32) - recurrenceMatrix)

    # first step
    # we take the two first possible values in the graph (top left), blank or first label (:2)
    # we put thaht in the matrix alpha_t of size (batch, 2U+1)
    alpha_t = tf.concat((pred[0, :, :2], tf.ones([batch, 2 * U + 1 - 2], dtype=tf.float32) * eps_nan),
                        axis=1)  # (batch, 2U+1)
    # we put this value inside the probability matrix of size (T, batch, 2U+1), at pos 0
    probability = tf.zeros([Time, batch, length], tf.float32)  # (1, batch, 2U+1)
    probability = tf.tensor_scatter_nd_update(probability, [[0]], alpha_t[tf.newaxis])

    # this is the main loop, using dynamic programming to compute the "alpha" of the CTC for all the path
    # this fill the probability matrix of size (T, batch, 2U+1)
    # at the end, we need only the bottom right value of the probability matrix, which is the probability of the paths
    # that lead to correct paths.
    def do(t, alpha_t, probability):
        # alpha t+1 is the sum of all possible previous alpha of possible path (defined with the recurrence matrix)
        # multiplied by the probability of the current step
        # here we convert to log relations, this lead to this formula:
        alpha_t = log_batch_dot(alpha_t, recurrence_relation[:, t - 1]) + pred[t, :, :]
        # add the alpha_t to the whole probability matrix
        probability = tf.tensor_scatter_nd_update(probability, [[t]], alpha_t[tf.newaxis])
        return t + 1, alpha_t, probability

    i = tf.constant(1)
    i, alpha_t, probability = tf.while_loop(lambda i, at, a: i < Time, do,
                                            [i, alpha_t, probability])

    # labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
    # labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
    probability = tf.transpose(probability, [1, 0, 2])
    # tf.print("proba , ", probability, summarize = -1)
    # pred len : [batch,1]
    # tf.print("token , pred , len", token_len, pred_len,summarize = -1)   # print(pred_len)
    probability = tf.squeeze(tf.gather(probability, pred_len - 1, axis=1, batch_dims=1), axis=1)
    # probability = probability[Time - 1, :, :]  # last true elem
    # token_len : [batch,1]
    realTokenlen = (token_len * 2 + 1)
    labels_2 = tf.squeeze(tf.gather(probability, realTokenlen - 2, axis=1, batch_dims=1), axis=1)
    # labels_2 = probability[:, length-2]#last true elem
    labels_1 = tf.squeeze(tf.gather(probability, realTokenlen - 1, axis=1, batch_dims=1), axis=1)
    # labels_1 = probability[:, length-1] #blank

    # labels_2 = tf.repeat(labels_2[tf.newaxis,:],batch,axis=0) #last true elem
    # labels_1 = tf.repeat(labels_1[tf.newaxis,:],batch,axis=0) #last true elem
    # labels_1 = tf.repeat(probability[token_len - 1,tf.newaxis, :, length-1],batch,axis=0)#last true elem
    # tf.print("labels_2", tf.shape(labels_2))
    # tf.print("labels_1", tf.shape(labels_1))

    labels_prob = log_sum_exp(labels_2, labels_1)
    #     pdb.set_trace()
    #     tf.print("labels_prob",labels_prob,summarize = -1)
    cost = -labels_prob
    return cost
@tf.function
def safe_log(x, eps=1e-5):
  """Avoid taking the log of a non-positive number."""
  safe_x = tf.where(x <= 0.0, eps, x)
  return tf.math.log(safe_x)
@tf.function
def ctc_loss_log_custom_prior(pred, token, pred_len, token_len, recurrenceMatrix, weightPrior=1.0):
    '''
    this is the same as ctc_loss_log_custom, but with the weighted label prior
    :param pred: (batch,Time, voca_size+1)
    :param token: (batch, U)
    :param pred_len: (batch,1)
    :param token_len: (batch,1)
    :param recurrenceMatrix : (batch, 2U+1, 2U+1)
    '''
    pred = tf.transpose(pred, [1, 0, 2])  # ( Time,batch, voca_size+1)
    pred = tf.math.log_softmax(pred)
    sm = tf.exp(pred)
    avg_sm = tf.stop_gradient(tf.reduce_mean(sm, axis=0, keepdims=True))  # (1,1,dim)
    pred = pred - safe_log(avg_sm)*weightPrior
    Time, batch = tf.shape(pred)[0], tf.shape(pred)[1]
    U = tf.shape(token)[1]
    eps_nan = -1e8

    # token_with_blank
    token_with_blank = tf.concat((tf.zeros([batch, U, 1], dtype=tf.int32), tf.cast(token[:, :, tf.newaxis], tf.int32)),
                                 axis=2)  # (batch, U,2)
    token_with_blank = tf.reshape(token_with_blank, [batch, -1])  # (batch, 2*U)
    # add a blank at the end of elems
    token_with_blank = tf.concat((token_with_blank, tf.zeros([batch, 1], dtype=tf.int32)), axis=1)  # (batch, 2U+1)
    # token_with_blank: [blank, index_e1,blank, index_e2, .... eU, blank]
    # replace all -1 by zeros
    token_with_blank = tf.where(token_with_blank == -1, tf.zeros_like(token_with_blank), token_with_blank)

    length = tf.shape(token_with_blank)[1]  # 2U+1

    pred = tf.gather(pred, tf.repeat(token_with_blank[tf.newaxis, :, :], Time, axis=0), axis=2,
                     batch_dims=2)  # (T, batch, 2U+1)]

    # recurrence relation
    # sec_diag = T.cat((T.zeros((batch, 2)).type(floatX), T.ne(token_with_blank[:, :-2], token_with_blank[:, 2:]).type(floatX)), dim=1) * T.ne(token_with_blank, blank).type(floatX)	# (batch, 2U+1)
    # recurrence_relation = (m_eye(length) + m_eye(length, k=1)).repeat(batch, 1, 1) + m_eye(length, k=2).repeat(batch, 1, 1) * sec_diag[:, None, :]	# (batch, 2U+1, 2U+1)
    recurrenceMatrix = tf.where(recurrenceMatrix != -1, recurrenceMatrix, tf.zeros_like(recurrenceMatrix, tf.float32))
    # tf.print("recu matrix , ", recurrenceMatrix, summarize = -1)
    recurrence_relation = eps_nan * (tf.ones_like(recurrenceMatrix, tf.float32) - recurrenceMatrix)

    # alpha
    alpha_t = tf.concat((pred[0, :, :2], tf.ones([batch, 2 * U + 1 - 2], dtype=tf.float32) * eps_nan),
                        axis=1)  # (batch, 2U+1)
    probability = tf.zeros([Time, batch, length], tf.float32)  # (1, batch, 2U+1)
    probability = tf.tensor_scatter_nd_update(probability, [[0]], alpha_t[tf.newaxis])

    # dynamic programming
    # (T, batch, 2U+1)
    # for t in T.arange(1, Time).type(longX):
    #     alpha_t = log_batch_dot(alpha_t, recurrence_relation) + pred[t]
    #     probability = T.cat((probability, alpha_t[None]), dim=0)

    def do(t, alpha_t, probability):
        alpha_t = log_batch_dot(alpha_t, recurrence_relation[:, t - 1]) + pred[t, :, :]
        # beta_t = log_sum_exp(log_batch_dot(beta_t, recurrence_relation[:,t]) + pred[t, :, :],
        #                      tf.math.log(-pred[t] + eps) + alpha_t)
        probability = tf.tensor_scatter_nd_update(probability, [[t]], alpha_t[tf.newaxis])
        # betas = tf.tensor_scatter_nd_update(betas, [[t]], beta_t[tf.newaxis])
        return t + 1, alpha_t, probability

    i = tf.constant(1)
    i, alpha_t, probability = tf.while_loop(lambda i, at, a: i < Time, do,
                                            [i, alpha_t, probability])

    # labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
    # labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
    probability = tf.transpose(probability, [1, 0, 2])
    # tf.print("proba , ", probability, summarize = -1)
    # pred len : [batch,1]
    # tf.print("token , pred , len", token_len, pred_len,summarize = -1)   # print(pred_len)
    probability = tf.squeeze(tf.gather(probability, pred_len - 1, axis=1, batch_dims=1), axis=1)
    # probability = probability[Time - 1, :, :]  # last true elem
    # token_len : [batch,1]
    realTokenlen = (token_len * 2 + 1)
    realTokenlenLess2 = realTokenlen - 2
    realTokenlenLess1 = realTokenlen - 1
    # ensure that the token is minimum 0
    realTokenlenLess2 = tf.where(realTokenlenLess2 < 0, tf.zeros_like(realTokenlenLess2), realTokenlenLess2)
    realTokenlenLess1 = tf.where(realTokenlenLess1 < 0, tf.zeros_like(realTokenlenLess1), realTokenlenLess1)




    labels_2 = tf.squeeze(tf.gather(probability, realTokenlenLess2, axis=1, batch_dims=1), axis=1)
    # labels_2 = probability[:, length-2]#last true elem
    labels_1 = tf.squeeze(tf.gather(probability, realTokenlenLess1, axis=1, batch_dims=1), axis=1)
    # labels_1 = probability[:, length-1] #blank

    # labels_2 = tf.repeat(labels_2[tf.newaxis,:],batch,axis=0) #last true elem
    # labels_1 = tf.repeat(labels_1[tf.newaxis,:],batch,axis=0) #last true elem
    # labels_1 = tf.repeat(probability[token_len - 1,tf.newaxis, :, length-1],batch,axis=0)#last true elem
    # tf.print("labels_2", tf.shape(labels_2))
    # tf.print("labels_1", tf.shape(labels_1))
    # tf.print("labels_2", labels_2)
    # tf.print("labels_1", labels_1)

    labels_prob = log_sum_exp(labels_2, labels_1)
    #     pdb.set_trace()
    # tf.print("labels_prob",labels_prob,summarize = -1)
    cost = -labels_prob
    return cost

@tf.function
def ctc_loss_log_normal(pred, token, pred_len, token_len, prior,weightPrior=1.0):
    '''
    this is the normal CTC (without guide) but with weighted label prior.
    :param pred: (batch,Time, voca_size+1)
    :param token: (batch, U)
    :param pred_len: (batch,1)
    :param token_len: (batch,1)
    :param recurrenceMatrix : (batch, 2U+1, 2U+1)
    '''
    blank=0
    pred = tf.transpose(pred, [1, 0, 2])  # ( Time,batch, voca_size+1)
    pred = tf.math.log_softmax(pred)
    if prior and weightPrior > 0:
        sm = tf.exp(pred)
        avg_sm = tf.stop_gradient(tf.reduce_mean(sm, axis=0, keepdims=True))  # (1,1,dim)
        pred = pred - safe_log(avg_sm)*weightPrior
    Time, batch = tf.shape(pred)[0], tf.shape(pred)[1]
    U = tf.shape(token)[1]
    eps_nan = -1e8

    # token_with_blank
    # "view" is a kind of reshape with sharing memory, but here the original concat is not kept, so same as reshape i guess
    token_with_blank = tf.concat((tf.zeros([batch, U, 1], dtype=tf.int32), tf.cast(token[:, :, tf.newaxis], tf.int32)),
                                 axis=2)  # (batch, U,2)
    token_with_blank = tf.reshape(token_with_blank, [batch, -1])  # (batch, 2*U)
    # add a blank at the end of elems
    token_with_blank = tf.concat((token_with_blank, tf.zeros([batch, 1], dtype=tf.int32)), axis=1)  # (batch, 2U+1)
    # token_with_blank: [blank, index_e1,blank, index_e2, .... eU, blank]
    token_with_blank = tf.where(token_with_blank == -1, tf.zeros_like(token_with_blank), token_with_blank)

    length = tf.shape(token_with_blank)[1]  # 2U+1

    pred = tf.gather(pred, tf.repeat(token_with_blank[tf.newaxis, :, :], Time, axis=0), axis=2,
                     batch_dims=2)  # (T, batch, 2U+1)]

    # recurrence relation
    consecutiveDifferent = tf.cast(tf.not_equal(token_with_blank[:, :-2], token_with_blank[:, 2:]), tf.float32)

    # pad with two blanks on the left
    consecutiveDifferent = tf.concat((tf.zeros((batch, 2), dtype=tf.float32), consecutiveDifferent), axis=1)

    # elements not blank in the GT (one on two)
    notBlanksToken = tf.cast(tf.not_equal(token_with_blank, blank), tf.float32)
    # kind of mask : True = Not consecutive identic elements, False = consecutive identic elements: can skip the blank,
    sec_diag = consecutiveDifferent * notBlanksToken  # (batch, 2U+1)

    # m_eye : identity matrix for k=0, ones of are shifted by k

    # recurrence_relation : (batch, 2U+1, 2U+1)
    recurrence_relation = \
        tf.repeat((tf.eye(length) + m_eye(length, k=1))[tf.newaxis], repeats=batch, axis=0) + \
        tf.repeat(m_eye(length, k=2)[tf.newaxis], repeats=batch, axis=0) * sec_diag[:, tf.newaxis, :]
    # tf.print("recurrence_relation\n",recurrence_relation,summarize=-1)
    recurrence_relation = eps_nan * (tf.ones_like(recurrence_relation) - recurrence_relation)

    # alpha
    alpha_t = tf.concat((pred[0, :, :2], tf.ones([batch, 2 * U + 1 - 2], dtype=tf.float32) * eps_nan),
                        axis=1)  # (batch, 2U+1)
    probability = tf.zeros([Time, batch, length], tf.float32)  # (1, batch, 2U+1)
    probability = tf.tensor_scatter_nd_update(probability, [[0]], alpha_t[tf.newaxis])

    # dynamic programming
    # (T, batch, 2U+1)
    # for t in T.arange(1, Time).type(longX):
    #     alpha_t = log_batch_dot(alpha_t, recurrence_relation) + pred[t]
    #     probability = T.cat((probability, alpha_t[None]), dim=0)

    def do(t, alpha_t, probability):
        alpha_t = log_batch_dot(alpha_t, recurrence_relation) + pred[t, :, :]
        # beta_t = log_sum_exp(log_batch_dot(beta_t, recurrence_relation[:,t]) + pred[t, :, :],
        #                      tf.math.log(-pred[t] + eps) + alpha_t)
        probability = tf.tensor_scatter_nd_update(probability, [[t]], alpha_t[tf.newaxis])
        # betas = tf.tensor_scatter_nd_update(betas, [[t]], beta_t[tf.newaxis])
        return t + 1, alpha_t, probability

    i = tf.constant(1)
    i, alpha_t, probability = tf.while_loop(lambda i, at, a: i < Time, do,
                                            [i, alpha_t, probability])

    # labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
    # labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
    probability = tf.transpose(probability, [1, 0, 2])
    # tf.print("proba , ", probability, summarize = -1)
    # pred len : [batch,1]
    # tf.print("token , pred , len", token_len, pred_len,summarize = -1)   # print(pred_len)
    probability = tf.squeeze(tf.gather(probability, pred_len - 1, axis=1, batch_dims=1), axis=1)
    # probability = probability[Time - 1, :, :]  # last true elem
    # token_len : [batch,1]
    realTokenlen = (token_len * 2 + 1)
    realTokenlenLess2 = realTokenlen - 2
    realTokenlenLess1 = realTokenlen - 1
    # ensure that the token is minimum 0
    realTokenlenLess2 = tf.where(realTokenlenLess2 < 0, tf.zeros_like(realTokenlenLess2), realTokenlenLess2)
    realTokenlenLess1 = tf.where(realTokenlenLess1 < 0, tf.zeros_like(realTokenlenLess1), realTokenlenLess1)

    labels_2 = tf.squeeze(tf.gather(probability, realTokenlenLess2, axis=1, batch_dims=1), axis=1)
    # labels_2 = probability[:, length-2]#last true elem
    labels_1 = tf.squeeze(tf.gather(probability, realTokenlenLess1, axis=1, batch_dims=1), axis=1)
    # labels_1 = probability[:, length-1] #blank

    # labels_2 = tf.repeat(labels_2[tf.newaxis,:],batch,axis=0) #last true elem
    # labels_1 = tf.repeat(labels_1[tf.newaxis,:],batch,axis=0) #last true elem
    # labels_1 = tf.repeat(probability[token_len - 1,tf.newaxis, :, length-1],batch,axis=0)#last true elem
    # tf.print("labels_2", tf.shape(labels_2))
    # tf.print("labels_1", tf.shape(labels_1))

    labels_prob = log_sum_exp(labels_2, labels_1)
    #     pdb.set_trace()
    tf.print("labels_prob",labels_prob,summarize = -1)
    cost = -labels_prob
    return cost
