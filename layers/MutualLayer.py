import tensorflow as tf


class MultualLayer(tf.layers.Layer):
    def __init__(self,labels=None, activation=None, regularizer=None):
        '''
        '''
        super(MultualLayer, self).__init__()
        self.activation = activation
        self.l2_reg = regularizer
        self.initializer = tf.orthogonal_initializer()

    def build(self, input_shape):
        '''
        '''
        self.d1 = self.d2 = int(input_shape[-1].value / 2)
        self.built = True

    def call(self, inputs):
        '''

        '''

        inputs_t1 = inputs[:, 0:self.d1]
        inputs_t2 = inputs[:, self.d1:]

        t1_res = tf.layers.dense(inputs=inputs_t1,
                                 units=self.d1,
                                 activation=self.activation,
                                 kernel_regularizer=self.l2_reg,
                                 kernel_initializer=self.initializer,
                                 name='Res_t1')
        t2_res = tf.layers.dense(inputs=inputs_t2,
                                 units=self.d2,
                                 activation=self.activation,
                                 kernel_regularizer=self.l2_reg,
                                 kernel_initializer=self.initializer,
                                 name='Res_t2')

        if self.weights is not None:
            weights = tf.expand_dims(tf.cast(self.weights, tf.float32),
                                     axis=1,
                                     name='weights')
            t1_mul = tf.multiply(t1_res, weights)
            t2_mul = tf.multiply(t2_res, weights)

        outputs_t1 = tf.add(inputs_t1, t2_mul, name='outputs_t1')
        outputs_t2 = tf.add(inputs_t2, t1_mul, name='outputs_t2')

        return outputs_t1, outputs_t2
