from __future__ import print_function

# import theano.tensor as T
import tensorflow as tf
import numpy as np

"""
    The main loss function (inner_cca_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
"""

def inner_cca_objective(y_pred, use_all_singular_values=True, outdim_size=128):
    """
    It is the loss function of CCA as introduced in the original paper. There can be other formulations.
    It is implemented on Tensorflow based on github@VahidooX's cca loss on Theano.
    y_true is just ignored
    """
    #  0< r1 <= 1e4*eps
    r1 = 1e-10  
    r2 = 1e-10
    eps = 0
    o1 = o2 = int(y_pred.shape[1] // 2)

    # unpack (separate) the output of networks for view 1 and view 2
    H1 = tf.transpose(y_pred[:, 0:o1])
    H2 = tf.transpose(y_pred[:, o1:o1 + o2])

    m = tf.shape(H1)[1]

    H1bar = H1 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H1, tf.ones([m, m]))
    H2bar = H2 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H2, tf.ones([m, m]))

    #tf.summary.scalar('H1_min', tf.reduce_min(tf.shape(H1)[0]))
    #tf.summary.scalar('H1_max', tf.reduce_max(m))

    SigmaHat12 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H1bar, H2bar, transpose_b=True)
    SigmaHat11 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H1bar, H1bar, transpose_b=True) + r1 * tf.eye(o1)
    SigmaHat22 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H2bar, H2bar, transpose_b=True) + r2 * tf.eye(o2)

    #tf.summary.histogram('s11_hist', SigmaHat11)
    #tf.summary.image('s11_img', SigmaHat11)

    # Calculating the root inverse of covariance matrices by using eigen decomposition
    [D1, V1] = tf.linalg.eigh(SigmaHat11)
    [D2, V2] = tf.linalg.eigh(SigmaHat22)  # Added to increase stability

    #tf.summary.scalar('D1_max', tf.reduce_max(D1))
    #tf.summary.scalar('D1_min', tf.reduce_min(D1))
    
    posInd1 = tf.where(tf.greater(D1, eps))
    D1 = tf.gather_nd(D1, posInd1)  # get eigen values that are larger than eps
    V1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(V1), tf.squeeze(posInd1)))

    #tf.summary.scalar('D1_shape', tf.shape(D1)[0])

    posInd2 = tf.where(tf.greater(D2, eps))
    D2 = tf.gather_nd(D2, posInd2)
    V2 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(V2), tf.squeeze(posInd2)))

    SigmaHat11RootInv = tf.matmul(tf.matmul(V1, tf.diag(D1 ** -0.5)), V1, transpose_b=True, name='SigmaHat11RootInv')
    SigmaHat22RootInv = tf.matmul(tf.matmul(V2, tf.diag(D2 ** -0.5)), V2, transpose_b=True, name='SigmaHat22RootInv')
    
    #tf.summary.scalar('s11_max', tf.reduce_max(SigmaHat11RootInv))
    #tf.summary.scalar('s11_min', tf.reduce_min(SigmaHat11RootInv))

    Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv, name='Tval')

    if use_all_singular_values:
        corr = tf.sqrt(tf.trace(tf.matmul(Tval, Tval, transpose_a=True)))
        #tf.summary.scalar('D1_corr', corr)
    else:
        [U, V] = tf.self_adjoint_eig(tf.matmul(Tval, Tval, transpose_a=True))
        U = tf.gather_nd(U, tf.where(tf.greater(U, eps)))
        kk = tf.reshape(tf.cast(tf.shape(U), tf.int32), [])
        K = tf.minimum(kk, outdim_size)
        w, _ = tf.nn.top_k(U, k=K)
        corr = tf.reduce_sum(tf.sqrt(w))
    print(-corr)
    return -corr
