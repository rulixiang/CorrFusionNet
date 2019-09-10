# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:44:37 2019

@author: rulix
"""
import tensorflow as tf
import numpy as np
from losses import *

class model():
    
    def __init__(self, base_net='resnet50'):
        if base_net is 'vgg16':
            self.base_net_t1=tf.keras.applications.VGG16
            self.base_net_t2=tf.keras.applications.VGG16
        elif base_net is 'vgg19':
            self.base_net_t1=tf.keras.applications.VGG19
            self.base_net_t2=tf.keras.applications.VGG19
        elif base_net is 'resnet50':
            self.base_net_t1=tf.keras.applications.ResNet50
            #self.base_net_t2=tf.keras.applications.ResNet50
        elif base_net is 'densenet121':
            self.base_net_t1=tf.keras.applications.DenseNet121
            self.base_net_t2=tf.keras.applications.DenseNet121
        elif base_net is 'densenet169':
            self.base_net_t1=tf.keras.applications.DenseNet169
            self.base_net_t2=tf.keras.applications.DenseNet169
        elif base_net is 'resnet34':
            self.base_net_t1=tf.keras.applications.ResNet50
            self.base_net_t2=tf.keras.applications.ResNet50

    def forward(self, inputs_t1=None, inputs_t2=None, label_t1=None, label_t2=None, num_classes=None):

        activation = tf.nn.tanh
        hidden_num = 1024
        l2_reg = tf.contrib.layers.l2_regularizer(1e-3)

        label_t1_onehot = tf.one_hot(indices=label_t1, depth=num_classes, name='label_t1_onehot')
        label_t2_onehot = tf.one_hot(indices=label_t2, depth=num_classes, name='label_t2_onehot')

        #with tf.variable_scope('conv_layers') as scope:
        with tf.device('/gpu:0'):
            conv_t1 = self.base_net_t1(weights=None, include_top=False,input_tensor=inputs_t1).output
        with tf.device('/gpu:0'):
            conv_t2 = self.base_net_t1(weights=None, include_top=False,input_tensor=inputs_t2).output
        
        flat_feature_t1 = tf.layers.flatten(inputs=conv_t1,name='flat_feature_t1')
        flat_feature_t2 = tf.layers.flatten(inputs=conv_t2,name='flat_feature_t2')

        with tf.name_scope('view_t1') as scope:
            with tf.device('/gpu:0'):
                dense1_t1 = tf.layers.dense(inputs=flat_feature_t1, units=hidden_num, activation=activation, kernel_regularizer=l2_reg, name='dense1_t1')
                dense2_t1 = tf.layers.dense(inputs=dense1_t1, units=hidden_num, activation=activation, kernel_regularizer=l2_reg, name='dense2_t1')
                dense3_t1 = tf.layers.dense(inputs=dense2_t1, units=hidden_num, activation=activation, kernel_regularizer=l2_reg, name='dense3_t1')
                #logits_t1 = tf.squeeze(logits_t1)

        with tf.name_scope('view_t2') as scope:
            with tf.device('/gpu:0'):
                dense1_t2 = tf.layers.dense(inputs=flat_feature_t2, units=hidden_num, activation=activation, kernel_regularizer=l2_reg, name='dense1_t2')
                dense2_t2 = tf.layers.dense(inputs=dense1_t2, units=hidden_num, activation=activation, kernel_regularizer=l2_reg, name='dense2_t2')
                dense3_t2 = tf.layers.dense(inputs=dense2_t2, units=hidden_num, activation=activation, kernel_regularizer=l2_reg, name='dense3_t2')
                #logits_t2 = tf.squeeze(logits_t2)
        ####################################
        cca_input_t1 = tf.layers.dense(inputs=dense2_t1, units=8, kernel_regularizer=l2_reg, name='cca_input_t1')
        cca_input_t2 = tf.layers.dense(inputs=dense2_t2, units=8, kernel_regularizer=l2_reg, name='cca_input_t2')
        cca_input = tf.concat([cca_input_t1, cca_input_t2], axis=1, name='cca_input')

        #with tf.variable_scope('cca_loss'):    
            #dcca_loss = deepcca_loss(cca_input_t1, cca_input_t2, outdim=4)
        ####################################
        logits_t1 = tf.layers.dense(inputs=dense3_t1, units=num_classes, kernel_regularizer=l2_reg, name='logits_t1')
        logits_t2 = tf.layers.dense(inputs=dense3_t2, units=num_classes, kernel_regularizer=l2_reg, name='logits_t2')

        self.predictions = {'classes_t1': tf.argmax(input=logits_t1, axis=1, name='classes_t1'), 
                            'classes_t2': tf.argmax(input=logits_t2, axis=1, name='classes_t2'), 
                            'probs_t1': tf.nn.softmax(logits=logits_t1, name='probs_t1'), 
                            'probs_t2': tf.nn.softmax(logits=logits_t2, name='probs_t2')}
        #print(np.shape(logits_t1))
        self.loss_t1 = tf.losses.softmax_cross_entropy(onehot_labels=label_t1_onehot,logits=logits_t1)
        self.loss_t2 = tf.losses.softmax_cross_entropy(onehot_labels=label_t2_onehot,logits=logits_t2)
        
        self.losses = self.loss_t1 + self.loss_t2# + 1e-2 * dcca_loss# + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        '''
        self.metrics = {'acc_t1': tf.metrics.accuracy(label_t1, predictions=self.predictions['classes_t1'])[0],
                        'acc_t2': tf.metrics.accuracy(label_t2, predictions=self.predictions['classes_t2'])[0]}
        '''
        with tf.name_scope('metrics') as scope:
            self.metrics_t1, self.metrics_t1_op = tf.metrics.accuracy(label_t1, predictions=self.predictions['classes_t1'],name='metrics_t1')
            self.metrics_t2, self.metrics_t2_op = tf.metrics.accuracy(label_t2, predictions=self.predictions['classes_t2'],name='metrics_t2')
        #running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        self.local_init = tf.local_variables_initializer()
        #for vars in tf.local_variables():
        #   print(vars) 
        tf.summary.scalar(name='losses/t1', tensor=self.loss_t1)
        tf.summary.scalar(name='losses/t2', tensor=self.loss_t2)
        tf.summary.scalar(name='losses/sum', tensor=self.losses)
        tf.summary.scalar(name='acc/t1', tensor=self.metrics_t1)
        tf.summary.scalar(name='acc/t2', tensor=self.metrics_t2)
        return True
