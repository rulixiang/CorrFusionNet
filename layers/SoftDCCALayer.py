import tensorflow as tf


class SoftDCCALayer(tf.layers.Layer):

    def __init__(self, labels=None, rho=0.9, lambda1=1e-1, lambda2=1e-4):
        
        super(SoftDCCALayer, self).__init__()
        # rho
        self.rho = rho
        # 
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.labels = labels

    def build(self, input_shape):
        # creat trainable weights
        self.d1 = self.d2 = int(input_shape[-1].value / 2)
        # initialize cov_t1, cov_t1
        self.cov_t1 = self.add_variable(
            name='cov_t1',
            shape=[self.d1, self.d1],
            initializer=tf.orthogonal_initializer(),
            trainable=False)
        self.cov_t2 = self.add_variable(
            name='cov_t2',
            shape=[self.d2, self.d2],
            initializer=tf.orthogonal_initializer(),
            trainable=False)

    def call(self, inputs):
        N = tf.shape(input=inputs)[0]

        view_t1 = inputs[:, 0:self.d1]
        view_t2 = inputs[:, self.d1:]
        
        if self.labels is not None:
            labels = tf.expand_dims(tf.cast(self.labels, tf.float32),
                                     axis=1,
                                     name='labels')
            view_t1 = tf.multiply(view_t1, labels)
            view_t2 = tf.multiply(view_t2, labels)
        
        corr_square = tf.square(tf.subtract(view_t1, view_t2, name='differ'))
        corr_loss = tf.sqrt(tf.reduce_sum(corr_square, axis=-1)+1e-8,name='corr_loss')

        self.cov_t1 = self.rho * self.cov_t1 + (1 - self.rho) * tf.matmul(view_t1, view_t1, transpose_a=True) / tf.cast(N - 1, tf.float32)
        self.cov_t2 = self.rho * self.cov_t2 + (1 - self.rho) * tf.matmul(view_t2, view_t2, transpose_a=True) / tf.cast(N - 1, tf.float32)

        decov_loss_t1 = tf.reduce_sum(tf.abs(self.cov_t1)) - tf.reduce_sum(
            tf.diag_part(self.cov_t1))
        decov_loss_t2 = tf.reduce_sum(tf.abs(self.cov_t2)) - tf.reduce_sum(
            tf.diag_part(self.cov_t2))

        decov_loss = tf.add(decov_loss_t1, decov_loss_t2)

        return self.lambda1 * tf.reduce_mean(corr_loss), self.lambda2 * tf.reduce_mean(decov_loss)
