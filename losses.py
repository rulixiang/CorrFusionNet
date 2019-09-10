import tensorflow as tf 
import numpy as np 

'''
view_t1 and view_t2 are centralized with a BN layer
'''

def softmax_loss(view_t1):
    '''
    original softmax
    '''

    return loss

def l_softmax_loss(view_t1):
    '''
    large margin softmax
    '''
    
    return loss

def softmax_loss(view_t1):
    '''
    arc face loss
    '''
    
    return loss

def sigmoid_loss(view_t1):
    '''
    sigmoid loss for binary classification
    '''

    return loss

def deepcca_loss(view_t1, view_t2, outdim=None):
    '''
    view_t1: nSamples x nBands
    view_t2: nSamples x nBands
    '''
    rcov1 = 1e-4
    rcov2 = 1e-4
    eps_eig = 1e-12
    N = tf.shape(input=view_t1)[0]
    d1 = d2 = tf.shape(input=view_t1)[1]

    # Remove mean.
    m1 = tf.reduce_mean(view_t1, axis=0, keep_dims=True)
    view_t1 = tf.subtract(view_t1, m1)

    m2 = tf.reduce_mean(view_t2, axis=0, keep_dims=True)
    view_t2 = tf.subtract(view_t2, m2)

    S11 = tf.matmul(tf.transpose(view_t1), view_t1) / (N-1) + rcov1 * tf.eye(d1)
    S22 = tf.matmul(tf.transpose(view_t2), view_t2) / (N-1) + rcov2 * tf.eye(d2)
    S12 = tf.matmul(tf.transpose(view_t1), view_t2) / (N-1)

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
    E3, U = tf.self_adjoint_eig(tf.matmul(T, tf.transpose(T)))
    idx3 = tf.where(E3 > eps_eig)[:, 0]
    # This is the thresholded rank.
    dim_svd = tf.cond(tf.size(idx3) < outdim, lambda: tf.size(idx3), lambda: outdim)

    loss = -tf.reduce_sum(tf.sqrt(E3[-outdim_svd:]))

    return loss


def decorr_loss(view_t1, sigma, rho=0.99):
    '''
    view_t1: nSamples x nBands
    view_t2: nSamples x nBands
    sigma: covariance matrix of last batch
    '''
    N = tf.shape(input=view_t1)[0]
    d = tf.shape(input=view_t1)[1]
    # convariance matrix for current batch
    S_curr = tf.matmul(tf.transpose(view_t1), view_t1) / (N-1)# + rcov1 * tf.eye(d)

    S = rho*sigma + (1-rho)*S_curr
    # TODO: test, the update of sigma 
    update_op = tf.assign(ref=sigma, value=S,)

    loss = tf.reduce_sum(tf.abs(S))-tf.reduce_sum(tf.diag(S))
    return loss, update_op

def softcca_loss(view_t1, view_t2):
    '''
    view_t1: nSamples x nBands, normlized with a BN layer
    view_t2: nSamples x nBands, normlized with a BN layer
    '''
    loss = tf.reduce_sum(tf.subtract(view_t1, view_t2, name='differ')**2)/2

    return loss