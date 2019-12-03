import tensorflow as tf


@tf.custom_gradient
def cca_loss(inputs, rcov1=1e-4, rcov2=1e-4, eps=1e-12):
    '''
    view_t1: nSamples x 2n Bands
    view_t2: nSamples x 2n Bands
    borrowed from https://bitbucket.org/qingming_tang/deep-canonical-correlation-analysis/
    '''
    eps_eig = eps
    N = tf.shape(input=inputs)[0]
    d1 = d2 = tf.cast(tf.shape(input=inputs)[1] / 2, tf.int32)

    view_t1 = inputs[:, 0:d1]
    view_t2 = inputs[:, d1:]

    # Remove mean.
    m1 = tf.reduce_mean(view_t1, axis=0, keep_dims=True)
    view_t1 = tf.subtract(view_t1, m1)

    m2 = tf.reduce_mean(view_t2, axis=0, keep_dims=True)
    view_t2 = tf.subtract(view_t2, m2)

    S11 = tf.matmul(tf.transpose(view_t1), view_t1) / tf.cast(
        N - 1, tf.float32) + rcov1 * tf.eye(d1)
    S22 = tf.matmul(tf.transpose(view_t2), view_t2) / tf.cast(
        N - 1, tf.float32) + rcov2 * tf.eye(d2)
    S12 = tf.matmul(tf.transpose(view_t1), view_t2) / tf.cast(
        N - 1, tf.float32)

    E1, V1 = tf.self_adjoint_eig(S11)
    E2, V2 = tf.self_adjoint_eig(S22)

    # For numerical stability.
    idx1 = tf.where(E1 > eps_eig)[:, 0]
    E1 = tf.gather(E1, idx1)
    V1 = tf.gather(V1, idx1, axis=1)

    idx2 = tf.where(E2 > eps_eig)[:, 0]
    E2 = tf.gather(E2, idx2)
    V2 = tf.gather(V2, idx2, axis=1)

    K11 = tf.matmul(tf.matmul(V1, tf.diag(tf.reciprocal(tf.sqrt(E1)))),
                    tf.transpose(V1))
    K22 = tf.matmul(tf.matmul(V2, tf.diag(tf.reciprocal(tf.sqrt(E2)))),
                    tf.transpose(V2))
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
        Delta11 = -0.5 * tf.matmul(M1, tf.matmul(SS, M1, transpose_b=True))
        Delta22 = -0.5 * tf.matmul(M2, tf.matmul(SS, M2, transpose_b=True))

        grad1 = 2 * tf.matmul(view_t1, Delta11) + tf.matmul(
            view_t2, Delta12, transpose_b=True)
        grad1 = grad1 / tf.cast(N - 1, tf.float32)

        grad2 = tf.matmul(view_t1, Delta12) + 2 * tf.matmul(view_t2, Delta22)
        grad2 = grad2 / tf.cast(N - 1, tf.float32)

        return tf.concat((grad1, grad2), axis=-1)

    return loss, grad


class CCALayer(tf.layers.Layer):
    def __init__(self):
        self.rcov1 = 1e-4
        self.rcov2 = 1e-4
        self.eps = 1e-12
        self.use_all_singular_values = True

    def build(self, input_shape):
        # creat trainable weigths
        self.built = True

    def call(self, inputs):

        return cca_loss(inputs=inputs,
                        rcov1=self.rcov1,
                        rcov2=self.rcov2,
                        eps=self.eps)
