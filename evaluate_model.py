import argparse
import logging
import os

import numpy as np
import scipy.io as sio
import tensorflow as tf

import utils

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', help='gpu device ID', default='0')
parser.add_argument('-m', '--model_dir', help='model directory',default='./VGG16-v1-dcca/model/0/')
parser.add_argument('-tst', '--tst_dir', help='testing file dir', default='./data_v1/tst/')
parser.add_argument('-val', '--val_dir', help='validation file dir', default='./data_v1/val/')
args = parser.parse_args()
bs = 16

def test(sess, graph, file_list=None):

    inputs_t1 = graph.get_tensor_by_name(name='inputs_t1:0')
    inputs_t2 = graph.get_tensor_by_name(name='inputs_t2:0')

    preds_t1 = graph.get_tensor_by_name(name='prediction_t1:0')
    preds_t2 = graph.get_tensor_by_name(name='prediction_t2:0')

    pred_label_t1 = None
    pred_label_t2 = None
    label_t1 = None
    label_t2 = None

    #xbatch1 = xbatch2 = np.zeros(shape=(5))

    for pfile in file_list:

        print('evaluating on file: %s......' % (pfile))
        xbatch1, xbatch2, ybatch1, ybatch2 = utils.LoadNpy(pfile)

        if label_t1 is None:
            label_t1 = ybatch1
            label_t2 = ybatch2
        else:
            label_t1 = np.concatenate((label_t1, ybatch1), axis=0)
            label_t2 = np.concatenate((label_t2, ybatch2), axis=0)

        for k1 in range(0, np.shape(xbatch1)[0], bs):
            lb = int(k1)
            ub = int(np.min((lb+bs, np.shape(xbatch1)[0])))
            lt1 = np.zeros_like(label_t1).astype(np.uint8)
            lt2 = np.zeros_like(label_t2).astype(np.uint8)
            # sess.run(base_model.local_init)
            feed_dict = {inputs_t1: xbatch1[lb:ub, :], inputs_t2: xbatch2[lb:ub, :]}
            tmp_pred_t1, tmp_pred_t2 = sess.run([preds_t1, preds_t2], feed_dict=feed_dict)

            if pred_label_t1 is None:
                pred_label_t1 = tmp_pred_t1
                pred_label_t2 = tmp_pred_t2
            else:
                pred_label_t1 = np.concatenate((pred_label_t1, tmp_pred_t1), axis=0)
                pred_label_t2 = np.concatenate((pred_label_t2, tmp_pred_t2), axis=0)

    pred_label_t1 = np.expand_dims(pred_label_t1, axis=1)
    pred_label_t2 = np.expand_dims(pred_label_t2, axis=1)

    return label_t1, label_t2, pred_label_t1, pred_label_t2


def main(args=None, val_file=None, tst_file=None):

    dir_list = os.listdir(args.model_dir)

    val_pred_t1 = None
    val_pred_t2 = None  
    tst_pred_t1 = None
    tst_pred_t2 = None

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model_name = args.model_dir + '/model.ckpt.meta'
    saver = tf.train.import_meta_graph(model_name)
    saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))

    graph = tf.get_default_graph()

    val_label_t1, val_label_t2, val_tmp_t1, val_tmp_t2 = test(sess=sess, graph=graph, file_list=val_file)
    tst_label_t1, tst_label_t2, tst_tmp_t1, tst_tmp_t2 = test(sess=sess, graph=graph, file_list=tst_file)
    val_acc_t1, val_acc_t2, val_acc_bi, val_acc_tr = utils.Accuracy(val_tmp_t1[:,0], val_tmp_t2[:,0], val_label_t1, val_label_t2)
    print('%s got val_acc_t1: %.4f, val_acc_t2: %.4f, val_acc_bi: %.4f, val_acc_tr: %.4f' % (model_name, val_acc_t1, val_acc_t2, val_acc_bi, val_acc_tr))
    tst_acc_t1, tst_acc_t2, tst_acc_bi, tst_acc_tr = utils.Accuracy(tst_tmp_t1[:,0], tst_tmp_t2[:,0], tst_label_t1, tst_label_t2)
    print('%s got tst_acc_t1: %.4f, tst_acc_t2: %.4f, tst_acc_bi: %.4f, tst_acc_tr: %.4f' % (model_name, tst_acc_t1, tst_acc_t2, tst_acc_bi, tst_acc_tr))
    '''
    sio.savemat('DenseNet121_pred.mat',
                mdict={
                    'val_pred_t1': val_tmp_t1,
                    'val_pred_t2': val_tmp_t2,
                    'tst_pred_t1': tst_tmp_t1,
                    'tst_pred_t2': tst_tmp_t2,
                    'val_label_t1': val_label_t1,
                    'val_label_t2': val_label_t2,
                    'tst_label_t1': tst_label_t1,
                    'tst_label_t2': tst_label_t2})
    '''
    tf.reset_default_graph()
    return None


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    val_list = os.listdir(args.val_dir)
    val_file = [args.val_dir+npz for npz in val_list]
    print(val_file)

    tst_list = os.listdir(args.tst_dir)
    tst_file = [args.tst_dir+npz for npz in tst_list]
    print(tst_file)

    main(args, val_file, tst_file)
