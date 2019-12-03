import math

import numpy as np
import tensorflow as tf


'''
view_t1 and view_t2 are centralized with a BN layer
'''

def Original_Softmax_loss(embeddings=None, weights=None, labels=None):
    """
    This is the orginal softmax loss, nothing to say
    """
    with tf.variable_scope("softmax"):
        
        logits = tf.matmul(embeddings, weights)
        pred_prob = tf.nn.softmax(logits=logits) # output probability
        # define cross entropy
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return pred_prob, loss

## implementation of DCCA with custom gradients
@tf.custom_gradient
def DCCA_loss(inputs):
    '''
    view_t1: nSamples x 2n Bands
    view_t2: nSamples x 2n Bands
    from https://bitbucket.org/qingming_tang/deep-canonical-correlation-analysis/
    '''
    #outdim = 16
    rcov1 = 1e-4
    rcov2 = 1e-4
    eps_eig = 1e-12
    N = tf.shape(input=inputs)[0]
    #d1 = d2 = tf.cast(tf.shape(input=inputs)[1]/2, tf.int32)
    d1 = d2 = int(inputs.shape[1].value/2)

    view_t1 = inputs[:,0:d1]
    view_t2 = inputs[:,d1:]

    # Remove mean.
    m1 = tf.reduce_mean(view_t1, axis=0, keep_dims=True)
    view_t1 = tf.subtract(view_t1, m1)

    m2 = tf.reduce_mean(view_t2, axis=0, keep_dims=True)
    view_t2 = tf.subtract(view_t2, m2)

    S11 = tf.matmul(tf.transpose(view_t1), view_t1) / tf.cast(N-1, tf.float32) + rcov1 * tf.eye(d1)
    S22 = tf.matmul(tf.transpose(view_t2), view_t2) / tf.cast(N-1, tf.float32) + rcov2 * tf.eye(d2)
    S12 = tf.matmul(tf.transpose(view_t1), view_t2) / tf.cast(N-1, tf.float32) 

    E1, V1 = tf.self_adjoint_eig(S11)
    E2, V2 = tf.self_adjoint_eig(S22)

    # For numerical stability.
    idx1 = tf.where(E1>eps_eig)[:,0]
    E1 = tf.gather(E1, idx1)
    V1 = tf.gather(V1, idx1, axis=1)

    idx2 = tf.where(E2>eps_eig)[:,0]
    E2 = tf.gather(E2, idx2)
    V2 = tf.gather(V2, idx2, axis=1)

    K11 = tf.matmul(tf.matmul(V1, tf.diag(tf.reciprocal(tf.sqrt(E1)))), tf.transpose(V1))
    K22 = tf.matmul(tf.matmul(V2, tf.diag(tf.reciprocal(tf.sqrt(E2)))), tf.transpose(V2))
    T = tf.matmul(tf.matmul(K11, S12), K22)

    # Eigenvalues are sorted in increasing order.
    E3, _ = tf.self_adjoint_eig(tf.matmul(T, tf.transpose(T)))
    idx3 = tf.where(E3 > eps_eig)[:, 0]
    # This is the thresholded rank.
    #dim_svd = tf.cond(tf.size(idx3) < outdim, lambda: tf.size(idx3), lambda: outdim)

    loss = -tf.reduce_sum(tf.sqrt(E3[-tf.size(idx3):]))

    def grad(dy):

        S, U, V = tf.linalg.svd(T, full_matrices=True, compute_uv=True)
        SS = tf.diag(S)
        ## 
        M1 = tf.matmul(K11, U)
        M2 = tf.matmul(K22, V)

        Delta12 = tf.matmul(M1, M2, transpose_b=True)
        Delta11 = -0.5*tf.matmul(M1, tf.matmul(SS, M1, transpose_b=True))
        Delta22 = -0.5*tf.matmul(M2, tf.matmul(SS, M2, transpose_b=True))
        
        grad1 = 2*tf.matmul(view_t1, Delta11)+tf.matmul(view_t2, Delta12, transpose_b=True)

        clip_val = 1e-1

        grad1 = grad1/tf.cast(N-1, tf.float32)
        grad1_clip = tf.clip_by_value(grad1, clip_value_max=clip_val, clip_value_min=-clip_val)
        grad2 = tf.matmul(view_t1, Delta12)+2*tf.matmul(view_t2, Delta22)
        grad2 = grad2/tf.cast(N-1, tf.float32)
        grad2_clip = tf.clip_by_value(grad2, clip_value_max=clip_val, clip_value_min=-clip_val)

        return tf.concat((grad1, grad2), axis=-1)

    return loss, grad

def Soft_DCCA_loss(view_t1=None, view_t2=None, eta1=1e-2, eta2=1e-3, rho=0.9):
    '''
    view_t1: nSamples by nBands, normlized by a BN layer
    view_t2: nSamples by nBands, normlized by a BN layer
    labels: nSamples vector, 
    [reference]: Scalable and Effective Deep CCA via Soft Decorrelation (CVPR18)
    '''

    #weights = tf.nn.l2_normalize(weights, axis=0)
    #view_t1 = tf.nn.l2_normalize(view_t1, axis=0)
    #view_t2 = tf.nn.l2_normalize(view_t2, axis=0)
    
    ## corr_loss: optional formulation of DCCA
    # weights = tf.cast(labels, tf.float32)
    # weights = tf.pow(-1., labels)
    corr_square = tf.square(tf.subtract(view_t1, view_t2,name='differ'))
    corr_loss = tf.sqrt(tf.reduce_sum(corr_square, axis=-1),name='corr_loss')

    ## decov loss
    N1 = tf.shape(input=view_t1)[0]
    d1 = tf.shape(input=view_t1)[1]

    N2 = tf.shape(input=view_t2)[0]
    d2 = tf.shape(input=view_t2)[1]


    S_t1 = tf.matmul(tf.transpose(view_t1), view_t1) / tf.cast(N1-1, tf.float32) 
    S1_init = tf.Variable(tf.zeros_like(S_t1), trainable=False, name='S1_init')
    S1_init = rho * S1_init + (1-rho)*S_t1
    S_t2 = tf.matmul(tf.transpose(view_t2), view_t2) / tf.cast(N2-1, tf.float32)
    S2_init = tf.Variable(tf.zeros_like(S_t2), trainable=False, name='S2_init')
    S2_init = rho * S2_init + (1-rho)*S_t2 

    decov_loss_t1 = tf.reduce_sum(tf.abs(S1_init))-tf.reduce_sum(tf.diag_part(S1_init))
    decov_loss_t2 = tf.reduce_sum(tf.abs(S2_init))-tf.reduce_sum(tf.diag_part(S2_init))
    decov_loss = decov_loss_t1 + decov_loss_t2

    #return corr_loss, eta*decov_loss
    return eta1*tf.reduce_mean(corr_loss), eta2*tf.reduce_mean(decov_loss)


