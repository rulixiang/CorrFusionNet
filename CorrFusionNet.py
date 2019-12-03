# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:44:37 2019

@author: rulix
"""
import numpy as np
import tensorflow as tf
from layers.FusionLayer import FusionLayer
from layers.SoftDCCALayer import SoftDCCALayer
from layers.CorrFusion import CorrFusion
from losses import *


class model():
    def __init__(self, inputs_shape=None):

        # alternative network: vgg16, vgg19, densenet...
        tf.reset_default_graph()
        self.base_net_t1 = tf.keras.applications.ResNet50
        self.base_net_t2 = tf.keras.applications.ResNet50

        self.inputs_t1 = tf.placeholder(dtype=tf.float32,
                                        shape=inputs_shape,
                                        name='inputs_t1')
        self.inputs_t2 = tf.placeholder(dtype=tf.float32,
                                        shape=inputs_shape,
                                        name='inputs_t2')
        self.labels_t1 = tf.placeholder(dtype=tf.uint8,
                                        shape=[None],
                                        name='labels_t1')
        self.labels_t2 = tf.placeholder(dtype=tf.uint8,
                                        shape=[None],
                                        name='labels_t2')

        self.activation = tf.nn.relu
        self.hidden_num = 1024
        self.l2_reg = tf.contrib.layers.l2_regularizer(1e-4)
        self.init = tf.glorot_normal_initializer()

    def forward(self, num_classes=None):

        label_t1_onehot = tf.one_hot(indices=self.labels_t1,
                                     depth=num_classes,
                                     name='label_t1_onehot')
        label_t2_onehot = tf.one_hot(indices=self.labels_t2,
                                     depth=num_classes,
                                     name='label_t2_onehot')
        label_bi = tf.equal(x=self.labels_t1,y=self.labels_t2,name='label_bi')
        

        with tf.variable_scope('conv_layers') as scope:
            
            conv_t1 = self.base_net_t1(weights=None,include_top=False,input_tensor=self.inputs_t1).output
            conv_t2 = self.base_net_t2(weights=None,include_top=False,input_tensor=self.inputs_t2).output

        flat_feature_t1 = tf.layers.flatten(inputs=conv_t1,
                                            name='flat_feature_t1')
        flat_feature_t2 = tf.layers.flatten(inputs=conv_t2,
                                            name='flat_feature_t2')

        ## dense layer1
        dense1_t1 = tf.layers.dense(inputs=flat_feature_t1,
                                    units=self.hidden_num,
                                    activation=self.activation,
                                    kernel_regularizer=self.l2_reg,
                                    kernel_initializer=self.init,
                                    name='dense1_t1')
        dense1_t2 = tf.layers.dense(inputs=flat_feature_t2,
                                    units=self.hidden_num,
                                    activation=self.activation,
                                    kernel_regularizer=self.l2_reg,
                                    kernel_initializer=self.init,
                                    name='dense1_t2')
        
        ## dense layer2
        dense2_t1 = tf.layers.dense(inputs=dense1_t1,
                                    units=self.hidden_num,
                                    activation=self.activation,
                                    kernel_regularizer=self.l2_reg,
                                    kernel_initializer=self.init,
                                    name='dense2_t1')
        dense2_t2 = tf.layers.dense(inputs=dense1_t2,
                                    units=self.hidden_num,
                                    activation=self.activation,
                                    kernel_regularizer=self.l2_reg,
                                    kernel_initializer=self.init,
                                    name='dense2_t2')
        
        ## dense layer3
        '''
        dense3_t1 = tf.layers.dense(inputs=dense2_t1,
                                    units=self.hidden_num,
                                    activation=self.activation,
                                    kernel_regularizer=self.l2_reg,
                                    kernel_initializer=self.init,
                                    name='dense3_t1')

        dense3_t2 = tf.layers.dense(inputs=dense2_t2,
                                    units=self.hidden_num,
                                    activation=self.activation,
                                    kernel_regularizer=self.l2_reg,
                                    kernel_initializer=self.init,
                                    name='dense3_t2')
        '''

        inputs_dim = dense2_t1.get_shape().as_list()[-1]
        with tf.name_scope('Residual_Fusion_Layer'):
            outputs_t1, outputs_t2, corr_loss, self.decov_loss, self.corr = CorrFusion(input_shape=inputs_dim).forward(inputs_t1=dense2_t1, inputs_t2=dense2_t2)

        self.corr_loss = tf.reduce_mean(tf.multiply(corr_loss, tf.cast(label_bi,tf.float32)), name='corr_loss')


        with tf.name_scope('losses') as scope:
            weights_t1 = tf.get_variable(
                name='weights_t1',
                shape=[outputs_t1.get_shape().as_list()[-1], num_classes],
                initializer=self.init)
            weights_t2 = tf.get_variable(
                name='weights_t2',
                shape=[outputs_t2.get_shape().as_list()[-1], num_classes],
                initializer=self.init)

            # Original_Softmax_loss
            self.pred_prob_t1, self.softmax_loss_t1 = Original_Softmax_loss(
                embeddings=outputs_t1, weights=weights_t1, labels=label_t1_onehot)
            self.pred_prob_t2, self.softmax_loss_t2 = Original_Softmax_loss(
                embeddings=outputs_t2, weights=weights_t2, labels=label_t2_onehot)

        ## original cnn
        #self.losses = self.softmax_loss_t1 + self.softmax_loss_t2
        ## cnn with DeepCCA
        #self.losses = self.softmax_loss_t1 + self.softmax_loss_t2 + 1e-3*self.dcca_loss# + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        ## cnn with softDCCA
        self.losses = self.softmax_loss_t1 + self.softmax_loss_t2 + self.corr_loss + self.decov_loss + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        self.prediction_t1 = tf.argmax(input=self.pred_prob_t1,
                                       axis=1,
                                       name='prediction_t1')
        self.prediction_t2 = tf.argmax(input=self.pred_prob_t2,
                                       axis=1,
                                       name='prediction_t2')

        with tf.name_scope('metrics') as scope:
            self.metrics_t1, self.metrics_t1_op = tf.metrics.accuracy(
                self.labels_t1,
                predictions=self.prediction_t1,
                name='metrics_t1')
            self.metrics_t2, self.metrics_t2_op = tf.metrics.accuracy(
                self.labels_t2,
                predictions=self.prediction_t2,
                name='metrics_t2')
        #running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        self.local_init = tf.local_variables_initializer()

        tf.summary.histogram(name='grads/t1',values=tf.gradients(self.softmax_loss_t1, outputs_t1))
        tf.summary.histogram(name='grads/t2',values=tf.gradients(self.softmax_loss_t2, outputs_t2))
        tf.summary.histogram(name='losses/corr_loss',values=corr_loss)
        tf.summary.histogram(name='losses/label_bi',values=tf.cast(label_bi,tf.float32))

        tf.summary.scalar(name='losses/t1', tensor=self.softmax_loss_t1)
        tf.summary.scalar(name='losses/t2', tensor=self.softmax_loss_t2)
        tf.summary.scalar(name='losses/sum', tensor=self.losses)
        tf.summary.scalar(name='losses/corr', tensor=self.corr)
        tf.summary.scalar(name='losses/decov_loss', tensor=self.decov_loss)
        tf.summary.scalar(name='losses/corr_loss', tensor=self.corr_loss)
        tf.summary.scalar(name='acc/t1', tensor=self.metrics_t1)
        tf.summary.scalar(name='acc/t2', tensor=self.metrics_t2)

        return True
