# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:42:26 2019

@author: rulix
"""

import argparse
import logging
import os
import time

import numpy as np
import scipy.io as sio
import tensorflow as tf

import utils
from model import model

logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', help='gpu device id', default='0')
parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=32)
parser.add_argument('-e', '--epoches', help='max epoches', type=int, default=50)
parser.add_argument('-t', '--tfboard', help='use tensorboadr', type=bool, default=False)
#parser.add_argument('-n', '--base_net', help='Backbone conv net', default='vgg16')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

trn_dir = './data_small/trn/'
tst_dir = './data_small/tst/'
val_dir = './data_small/val/'

epoches = args.epoches
batchsize = args.batch_size

num_classes = 14
base_net = 'resnet50'

def main(trn_file, val_file):

    inputs_t1 = tf.placeholder(dtype=tf.float32,shape=[None, 200, 200, 3],name='inputs_t1')
    inputs_t2 = tf.placeholder(dtype=tf.float32,shape=[None, 200, 200, 3],name='inputs_t2')
    label_t1 = tf.placeholder(dtype=tf.uint8, shape=[None], name='label_t1')
    label_t2 = tf.placeholder(dtype=tf.uint8, shape=[None], name='label_t2')

    base_model = model(base_net=base_net)
    '''
    '''
    global_steps = tf.Variable(0, trainable=False)
    base_model.forward(inputs_t1, inputs_t2, label_t1, label_t2, num_classes)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(base_model.losses, global_step=global_steps)
    initializer = tf.global_variables_initializer()
    
    sess.run(base_model.local_init)
    sess.run(initializer)
    
    if args.tfboard:
        graph_path = './cnn/tensorboard'
        result_path = './cnn/result'
        model_path = './cnn/model'
        writer_trn = tf.summary.FileWriter(logdir=graph_path+'/trn', graph=sess.graph)
        writer_val = tf.summary.FileWriter(logdir=graph_path+'/val', graph=sess.graph)
        summary_merge = tf.summary.merge_all()

    temp_acc_t1 = 0.
    temp_acc_t2 = 0.
        
    for e in range(epoches):
        
        ### optimization
        logging.info('Epoch %2d, training started......'%(e))
        for trn in trn_file:
            logging.info('Epoch %2d, training on file: %s......'%(e,trn))

            xtrn1, xtrn2, ytrn1, ytrn2 = utils.LoadNpy(trn)

            for k1 in range(0, np.shape(xtrn1)[0], batchsize):
                lb = int(k1)
                ub = int(np.min((lb+batchsize,np.shape(xtrn1)[0])))
                feed_dict = {inputs_t1:xtrn1[lb:ub,:], label_t1:ytrn2[lb:ub], inputs_t2:xtrn2[lb:ub,:], label_t2:ytrn2[lb:ub]}
                sess.run([optimizer], feed_dict=feed_dict)
        logging.info('Epoch %2d, training finished.....'%(e))
        
        ### training
        sess.run(base_model.local_init)
        logging.info('Epoch %2d, evaluating on training set......'%(e))
        for trn in trn_file:
            logging.info('Epoch %2d, evaluating on file: %s......'%(e,trn))

            xtrn1, xtrn2, ytrn1, ytrn2 = utils.LoadNpy(trn)

            for k1 in range(0, np.shape(xtrn1)[0], batchsize):
                lb = int(k1)
                ub = int(np.min((lb+batchsize,np.shape(xtrn1)[0])))

                #sess.run(base_model.local_init)
                feed_dict = {inputs_t1:xtrn1[lb:ub,:], label_t1:ytrn2[lb:ub], inputs_t2:xtrn2[lb:ub,:], label_t2:ytrn2[lb:ub]}
                sess.run([base_model.metrics_t1_op, base_model.metrics_t2_op], feed_dict=feed_dict)

        if args.tfboard:
            writer_trn.add_summary(sess.run(summary_merge, feed_dict=feed_dict),global_step=e)
            writer_trn.flush()

        acc_t1, acc_t2 = sess.run([base_model.metrics_t1, base_model.metrics_t2])
        logging.info('Epoch %2d, evaluating on training set finished, acc_t1 is: %.4f, acc_t2 is %.4f.....'%(e, acc_t1, acc_t2))
        

        ### validation
        sess.run(base_model.local_init)
        logging.info('Epoch %2d, evaluating on validation set started.....'%(e))
        for val in val_file:
            logging.info('Epoch %2d, evaluating on file %s.....'%(e, val))

            xval1, xval2, yval1, yval2 = utils.LoadNpy(val)

            for k1 in range(0, np.shape(xval1)[0], batchsize):
                lb = int(k1)
                ub = int(np.min((lb+batchsize,np.shape(xval1)[0])))

                #sess.run(base_model.local_init)
                feed_dict = {inputs_t1:xval1[lb:ub,:], label_t1:yval1[lb:ub], inputs_t2:xval2[lb:ub,:], label_t2:yval2[lb:ub]}
                sess.run([base_model.metrics_t1_op, base_model.metrics_t2_op], feed_dict=feed_dict)
                
        if args.tfboard: 
            writer_val.add_summary(sess.run(summary_merge, feed_dict=feed_dict),global_step=e)
            writer_val.flush()

        val_acc_t1, val_acc_t2 = sess.run([base_model.metrics_t1, base_model.metrics_t2])
        logging.info('Epoch %2d, evaluating on validation set finished, acc_t1 is: %.4f, acc_t2 is %.4f.....'%(e, val_acc_t1, val_acc_t2))
    
        #### save better model
        if args.tfboard & ((val_acc_t1+val_acc_t2) > (temp_acc_t1+temp_acc_t2)):
            model_name = 'cnn_%s_epoches=%2d_batchsize=%2d.ckpt'%(base_net,epoches,batchsize)
            logging.info('Saving model to %s......'%(model_name))
            saver = tf.train.Saver(max_to_keep=3,)
            saver.save(sess=sess,save_path=model_path+'/'+model_name)
            logging.info('Model saved......\n')
            temp_acc_t1 = val_acc_t1
            temp_acc_t2 = val_acc_t2
        else:
            logging.info('Performance is worse than the last epoch, don\'t save model....\n')

    #sio.savemat('test_result_step=100.mat', mdict={'pred_label_t1':pred_label_t1,'pred_label_t2':pred_label_t2,'test_label_t1':test_label_t1,'test_label_t2':test_label_t2})
    sess.close()
    return True


if __name__ == '__main__':

    trn_list = os.listdir(trn_dir)
    trn_file = [trn_dir+npz for npz in trn_list]
    print(trn_file)

    val_list = os.listdir(val_dir)
    val_file = [val_dir+npz for npz in val_list]
    print(val_file)

    main(trn_file, val_file)
