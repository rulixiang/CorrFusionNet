import tensorflow as tf


class SDCCALayer(tf.layers.Layer):

    def __init__(self, labels=None, rho=0.9, eta1=1e-2, eta2=1e-3):
        
        super(SDCCALayer, self).__init__()
        self.eta1 = eta1
        self.rho = rho
        self.eta2 = eta2
        self.weights = labels

    def build(self, input_shape):
        # creat trainable weigths

        self.d1 = self.d2 = int(input_shape[-1].value / 2)

        self.COV_t1 = self.add_variable(
            name='cov_t1',
            shape=[self.d1, self.d1],
            initializer=tf.orthogonal_initializer(),
            trainable=True)
        self.COV_t2 = self.add_variable(
            name='cov_t2',
            shape=[self.d2, self.d2],
            initializer=tf.orthogonal_initializer(),
            trainable=True)

    def call(self, inputs):
        N = tf.shape(input=inputs)[0]

        view_t1 = inputs[:, 0:self.d1]
        view_t2 = inputs[:, self.d1:]

        if self.weights is not None:
            weights = tf.expand_dims(tf.cast(self.weights, tf.float32),
                                     axis=1,
                                     name='weights')
            view_t1 = tf.multiply(view_t1, weights)
            view_t2 = tf.multiply(view_t2, weights)

        corr_square = tf.square(tf.subtract(view_t1, view_t2, name='differ'))
        corr_loss = tf.sqrt(tf.reduce_sum(corr_square, axis=-1),
                            name='corr_loss')

        self.COV_t1 = rho * self.COV_t1 + (1 - rho) * tf.matmul(
            view_t1, view_t1, transpose_a=True) / tf.cast(N - 1, tf.float32)
        self.COV_t2 = rho * self.COV_t2 + (1 - rho) * tf.matmul(
            view_t2, view_t2, transpose_a=True) / tf.cast(N - 1, tf.float32)

        decov_loss_t1 = tf.reduce_sum(tf.abs(self.COV_t1)) - tf.reduce_sum(
            tf.diag_part(self.COV_t1))
        decov_loss_t2 = tf.reduce_sum(tf.abs(self.COV_t2)) - tf.reduce_sum(
            tf.diag_part(self.COV_t2))

        decov_loss = tf.add(decov_loss_t1, decov_loss_t2)

        return eta1 * tf.reduce_mean(corr_loss), eta2 * tf.reduce_mean(
            decov_loss)
