import tensorflow as tf

class CorrFusion(object):

    def __init__(self, input_shape=None, rho=0.9, lambda1=1e-1, lambda2=1e-4):
    
        #super(SDCCALayer, self).__init__()
        # scale: dimensionality reduction ratio
        self.scale = 2
        self.eps = 1e-12
        # rho: momentum parameter
        self.rho = rho
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.d1 = self.d2 = int(input_shape)
        self.squezze_d1 = self.squezze_d2 = int(input_shape / self.scale)

        self.cov_t1 = tf.get_variable(name='cov_t1',shape=[self.squezze_d1, self.squezze_d1],initializer=tf.orthogonal_initializer(),trainable=False)
        self.cov_t2 = tf.get_variable(name='cov_t2',shape=[self.squezze_d2, self.squezze_d2],initializer=tf.orthogonal_initializer(),trainable=False)
        

    def canonical_corr(self, view_t1, view_t2):

        '''
        view_t1: nSamples x b Bands
        view_t2: nSamples x b Bands
        from https://bitbucket.org/qingming_tang/deep-canonical-correlation-analysis
        '''
        rcov1 = rcov2 = 1e-4
        N = tf.shape(input=view_t1)[0]
        d1 = d2 = tf.cast(tf.shape(input=view_t1)[1], tf.int32)
        
        # Remove mean.
        m1 = tf.reduce_mean(view_t1, axis=0, keep_dims=True)
        view_t1 = tf.subtract(view_t1, m1)
        m2 = tf.reduce_mean(view_t2, axis=0, keep_dims=True)
        view_t2 = tf.subtract(view_t2, m2)
        S11 = tf.matmul(tf.transpose(view_t1), view_t1) / tf.cast(N - 1, tf.float32) + rcov1 * tf.eye(d1)
        S22 = tf.matmul(tf.transpose(view_t2), view_t2) / tf.cast(N - 1, tf.float32) + rcov2 * tf.eye(d2)
        S12 = tf.matmul(tf.transpose(view_t1), view_t2) / tf.cast(N - 1, tf.float32)
        
        E1, V1 = tf.self_adjoint_eig(S11)
        E2, V2 = tf.self_adjoint_eig(S22)
        
        # For numerical stability.
        idx1 = tf.where(E1 > self.eps)[:, 0]
        E1 = tf.gather(E1, idx1)
        V1 = tf.gather(V1, idx1, axis=1)
        
        idx2 = tf.where(E2 > self.eps)[:, 0]
        E2 = tf.gather(E2, idx2)
        V2 = tf.gather(V2, idx2, axis=1)

        K11 = tf.matmul(tf.matmul(V1, tf.diag(tf.reciprocal(tf.sqrt(E1)))),tf.transpose(V1))
        K22 = tf.matmul(tf.matmul(V2, tf.diag(tf.reciprocal(tf.sqrt(E2)))),tf.transpose(V2))
        T = tf.matmul(tf.matmul(K11, S12), K22)

        corr = tf.sqrt(tf.trace(tf.matmul(T, tf.transpose(T))) + self.eps)
        return corr

    def forward(self, inputs_t1=None, inputs_t2=None):
        '''
        inputs_t1: nSamples x b Bands
        inputs_t2: nSamples x b Bands
        from https://bitbucket.org/qingming_tang/deep-canonical-correlation-analysis
        '''
        N = tf.shape(input=inputs_t1)[0]

        ## dimensionality reduction
        squezze_t1 = tf.layers.dense(inputs=inputs_t1, units=self.squezze_d1, activation=tf.nn.relu, name='squezze_t1')
        squezze_t2 = tf.layers.dense(inputs=inputs_t2, units=self.squezze_d2, activation=tf.nn.relu, name='squezze_t2')

        ## batch normlization
        bn_t1 = tf.layers.batch_normalization(inputs=squezze_t1,axis=-1,name='bn_t1')
        bn_t2 = tf.layers.batch_normalization(inputs=squezze_t2,axis=-1,name='bn_t2')

        ## dimensionality increasing
        excitation_t1 = tf.layers.dense(inputs=bn_t1, units=self.d1, activation=tf.nn.relu, name='excitation_t1')
        excitation_t2 = tf.layers.dense(inputs=bn_t2, units=self.d2, activation=tf.nn.relu, name='excitation_t2')

        ## compute canonical correlation
        with tf.name_scope('cca_corr') as scope:
            corr = self.canonical_corr(bn_t1,bn_t2)
        
        ## correlation loss, adding eps for numerical stability
        corr_square = tf.square(tf.subtract(bn_t1, bn_t2, name='differ')) + self.eps
        corr_loss = tf.sqrt(tf.reduce_sum(corr_square, axis=-1))

        self.cov_t1 = self.rho * self.cov_t1 + (1 - self.rho) * tf.matmul(bn_t1, bn_t1, transpose_a=True) / tf.cast(N - 1, tf.float32)
        self.cov_t2 = self.rho * self.cov_t2 + (1 - self.rho) * tf.matmul(bn_t2, bn_t2, transpose_a=True) / tf.cast(N - 1, tf.float32)

        ## compute decov loss
        with tf.name_scope('decov_loss'):
            ## compute the l1 de-cov loss
            decov_loss_t1 = tf.reduce_sum(tf.abs(self.cov_t1)) - tf.reduce_sum(tf.diag_part(self.cov_t1))
            decov_loss_t2 = tf.reduce_sum(tf.abs(self.cov_t2)) - tf.reduce_sum(tf.diag_part(self.cov_t2))
            decov_loss = self.lambda2 * tf.reduce_mean(tf.add(decov_loss_t1, decov_loss_t2), name='decov_loss')
        
        ## corr fusion block 
        with tf.name_scope('corr_fusion_block'):
            weights = 1 - tf.nn.tanh(tf.expand_dims(corr_loss, axis=1))
            weighted_t1 = tf.multiply(excitation_t1, weights, name='weighted_t1')
            weighted_t2 = tf.multiply(excitation_t2, weights, name='weighted_t2')

        corr_loss = self.lambda1 * corr_loss

        outputs_t1 = inputs_t1 + weighted_t2
        outputs_t2 = inputs_t2 + weighted_t1

        return outputs_t1, outputs_t2, corr_loss, decov_loss, corr
