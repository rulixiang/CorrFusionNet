# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:42:26 2019

@author: rulix
"""

import logging
import os
import time
from parser import argparser

import numpy as np
import scipy.io as sio
import tensorflow as tf

import utils
from ReSDCCANet import model

logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

def test(model=None, session=None, file_list=None, batch_size=None, use_tfboard=True, summary=None, tb_writer=None, step=0):
    session.run(model.local_init)
    for pfile in file_list:

        logging.info('Epoch %2d, evaluating on file: %s......'%(step,pfile))
        xbatch1, xbatch2, ybatch1, ybatch2 = utils.LoadNpy(pfile)

        for k1 in range(0, np.shape(xbatch1)[0], batch_size):
            lb = int(k1)
            ub = int(np.min((lb+batch_size,np.shape(xbatch1)[0])))
            #sess.run(base_model.local_init)
            feed_dict = {model.inputs_t1: xbatch1[lb:ub, :], model.labels_t1: ybatch1[lb:ub],
                         model.inputs_t2: xbatch2[lb:ub, :], model.labels_t2: ybatch2[lb:ub]}
            session.run([model.metrics_t1_op, model.metrics_t2_op], feed_dict=feed_dict)

    if args.use_tfboard is True:
        tb_writer.add_summary(session.run(summary, feed_dict=feed_dict),global_step=step)
        tb_writer.flush()

    acc_t1, acc_t2 = session.run([model.metrics_t1, model.metrics_t2])
    return acc_t1, acc_t2

def main(trn_file=None, val_file=None, tst_file=None, args=None):

    inputs_shape = [None, 200, 200, 3]
    base_model = model(inputs_shape=inputs_shape)
    base_model.forward(num_classes=args.num_classes)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    global_steps = tf.Variable(0, trainable=False)
    optimizer = tf.train.MomentumOptimizer(learning_rate=1e-2,momentum=0.9,use_nesterov=True).minimize(base_model.losses, global_step=global_steps)
    '''
    conv_vars = tf.trainable_variables(scope='conv_layers')
    dense_vars = tf.trainable_variables()[len(conv_vars):]
    optimizer_dcca = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(base_model.dcca_loss, var_list=dense_vars, global_step=global_steps)

    optimizer = tf.group(optimizer, optimizer_dcca)
    '''
    initializer = tf.global_variables_initializer()
    
    sess.run(base_model.local_init)
    sess.run(initializer)

    summary_merge = None
    writer_trn = writer_tst = writer_val = None
    if args.use_tfboard is True:
        writer_trn = tf.summary.FileWriter(logdir=args.tb_path+'/trn', graph=sess.graph)
        writer_val = tf.summary.FileWriter(logdir=args.tb_path+'/val', graph=sess.graph)
        writer_tst = tf.summary.FileWriter(logdir=args.tb_path+'/tst', graph=sess.graph)
        summary_merge = tf.summary.merge_all()

    temp_acc_t1 = 0.
    temp_acc_t2 = 0.
    
    f = open(args.log_path+'log.txt', 'w')
    cnt = 0
    for step in range(args.epoches):
        
        ### optimization
        logging.info('Epoch %2d, training started......'%(step))
        for trn in trn_file:
            logging.info('Epoch %2d, training on file: %s......'%(step,trn))

            xtrn1, xtrn2, ytrn1, ytrn2 = utils.LoadNpy(trn)

            for k1 in range(0, np.shape(xtrn1)[0], args.batch_size):
                lb = int(k1)
                ub = int(np.min((lb + args.batch_size,np.shape(xtrn1)[0])))
                feed_dict = {base_model.inputs_t1: xtrn1[lb:ub, :], base_model.labels_t1: ytrn1[lb:ub],
                             base_model.inputs_t2: xtrn2[lb:ub, :], base_model.labels_t2: ytrn2[lb:ub]}
                sess.run([optimizer], feed_dict=feed_dict)
                '''
                if True:
                    writer_trn.add_summary(sess.run(summary_merge, feed_dict=feed_dict),global_step=cnt)
                    cnt = cnt + 1
                    writer_trn.flush()
                '''
        logging.info('Epoch %2d, training finished.....'%(step))
        
        ### training
        logging.info('Epoch %2d, evaluating on training set......'%(step))
        f.writelines('Epoch %2d, evaluating on training set......\n'%(step))
        trn_acc_t1, trn_acc_t2 = test(model=base_model, session=sess, file_list=trn_file, batch_size=args.batch_size,
                                          use_tfboard=args.use_tfboard, summary=summary_merge, tb_writer=writer_trn, step=step)
        logging.info('Epoch %2d, evaluating on training set finished, acc_t1 is: %.4f, acc_t2 is %.4f.....'%(step, trn_acc_t1, trn_acc_t2))
        f.writelines('Epoch %2d, evaluating on training set finished, acc_t1 is: %.4f, acc_t2 is %.4f.....\n'%(step, trn_acc_t1, trn_acc_t2))

        ### validation
        logging.info('Epoch %2d, evaluating on validation set......'%(step))
        f.writelines('Epoch %2d, evaluating on validation set......\n'%(step))
        val_acc_t1, val_acc_t2 = test(model=base_model, session=sess, file_list=val_file, batch_size=args.batch_size,
                                          use_tfboard=args.use_tfboard, summary=summary_merge, tb_writer=writer_val, step=step)
        logging.info('Epoch %2d, evaluating on validation set finished, acc_t1 is: %.4f, acc_t2 is %.4f.....'%(step, val_acc_t1, val_acc_t2))
        f.writelines('Epoch %2d, evaluating on validation set finished, acc_t1 is: %.4f, acc_t2 is %.4f.....\n'%(step, val_acc_t1, val_acc_t2))

        ### testing
        logging.info('Epoch %2d, evaluating on testing set......'%(step))
        f.writelines('Epoch %2d, evaluating on testing set......\n'%(step))
        tst_acc_t1, tst_acc_t2 = test(model=base_model, session=sess, file_list=tst_file, batch_size=args.batch_size,
                                          use_tfboard=args.use_tfboard, summary=summary_merge, tb_writer=writer_tst, step=step)
        logging.info('Epoch %2d, evaluating on testing set finished, acc_t1 is: %.4f, acc_t2 is %.4f.....'%(step, tst_acc_t1, tst_acc_t2))
        f.writelines('Epoch %2d, evaluating on testing set finished, acc_t1 is: %.4f, acc_t2 is %.4f.....\n\n'%(step, tst_acc_t1, tst_acc_t2))

        #### save better model
        if args.save_model & ((val_acc_t1+val_acc_t2) > (temp_acc_t1+temp_acc_t2)):
            model_name = 'model.ckpt'
            logging.info('Saving model to %s......'%(args.model_path+'/'+model_name))
            saver = tf.train.Saver(max_to_keep=3,)
            saver.save(sess=sess,save_path=args.model_path+'/'+model_name)
            logging.info('Model saved......\n')
            temp_acc_t1 = val_acc_t1
            temp_acc_t2 = val_acc_t2
        else:
            logging.info('Performance is worse than the last epoch, don\'t save model....\n')
    f.close()
    sess.close()
    return True


if __name__ == '__main__':

    args = argparser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    trn_list = os.listdir(args.trn_dir)
    trn_file = [args.trn_dir+npz for npz in trn_list]
    print(trn_file)

    val_list = os.listdir(args.val_dir)
    val_file = [args.val_dir+npz for npz in val_list]
    print(val_file)

    tst_list = os.listdir(args.tst_dir)
    tst_file = [args.tst_dir+npz for npz in tst_list]
    print(tst_file)

    for k in range(10):
        args.log_path = './cnn-sdcca-res/log/'+str(k)+'/'
        args.model_path = './cnn-sdcca-res/model/'+str(k)+'/'
        
        if os.path.exists(args.log_path) is False:
            os.makedirs(args.log_path)
        if args.save_model and (os.path.exists(args.model_path) is False):
            os.makedirs(args.model_path)
        main(trn_file, val_file, tst_file, args)
