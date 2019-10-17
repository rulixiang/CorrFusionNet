# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:05:57 2019

@author: rulix
"""
import os
import logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(
    format='%(asctime)-15s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import sklearn

trn_dir = '../data2/trn/'
val_dir = '../data2/val/'
tst_dir = '../data2/tst/'

def LoadNpy(filename=None):

    npy = np.load(file=filename)
    image_t1 = npy['image_t1']
    image_t1 = image_t1.astype(np.float32)/255#-0.5
    image_t2 = npy['image_t2']
    image_t2 = image_t2.astype(np.float32)/255#-0.5
    label_t1 = npy['label_t1'] - 1
    label_t2 = npy['label_t2'] - 1

    return image_t1, image_t2, label_t1, label_t2
    
def kernel_hik(data_1, data_2):

    kernel = np.zeros((data_1.shape[0], data_2.shape[0]))

    for d in range(data_1.shape[1]):
        column_1 = data_1[:, d].reshape(-1, 1)
        column_2 = data_2[:, d].reshape(-1, 1)
        kernel += np.minimum(column_1, column_2.T)

    return kernel


def extract_patch(image=None, ksize=[1,10,10,1], stride=[1,5,5,1]):

    tf.reset_default_graph()
    '''
    ## for small dataset
    image_patch = tf.extract_image_patches(images=image, ksizes=ksize, strides=stride, rates=[1,1,1,1], padding='VALID')
    
    sess = tf.InteractiveSession()
    image_patch = image_patch.eval()
    sess.close()
    '''
    step = 1000
    shape = list(np.shape(image))
    dim1 = int(shape[1]/stride[1])
    dim2 = int(shape[2]/stride[2])
    image_patch = np.zeros(shape=(shape[0], dim1, dim2, ksize[1]*ksize[2]*3))
    shape[0] = None

    inputs = tf.placeholder(tf.float32, shape=shape)
    batch = tf.extract_image_patches(images=inputs, ksizes=ksize, strides=stride, rates=[1,1,1,1], padding='VALID')

    sess = tf.Session()

    for k1 in range(0, image_patch.shape[0], step):
        ub = np.min((image_patch.shape[0],k1+step))
        image_patch[k1:ub,:] = sess.run(batch, feed_dict={inputs:image[k1:ub,:]})

    sess.close()

    return image_patch

def to_histogram(data, dim):

    hist_feature = np.zeros((data.shape[0],dim))

    for k1 in range(data.shape[0]):
        vector = data[k1,:]
        unique_data, counts = np.unique(vector, return_counts=True)
        hist_feature[k1,unique_data] = counts

    return hist_feature

def extract_feature(file_list, ksize = 10, stride = 10):

    patch_t1 = None
    label_t1 = None
    patch_t2 = None
    label_t2 = None

    for file in file_list:

        image_t1, image_t2, label_t1, label_t2 = LoadNpy(file)

        temp_patch_t1 = extract_patch(image=image_t1, ksize=[1,ksize,ksize,1], stride=[1,stride,stride,1])
        temp_patch_t2 = extract_patch(image=image_t2, ksize=[1,ksize,ksize,1], stride=[1,stride,stride,1])

        if patch_t1 is None:
            patch_t1 = temp_patch_t1
            label_t1 = label_t1
            patch_t2 = temp_patch_t2
            label_t2 = label_t2
        else:
            patch_t1 = np.concatenate((patch_t1,temp_patch_t1),axis=0)
            label_t1 = np.concatenate((label_t1,label_t1),axis=0)
            patch_t2 = np.concatenate((patch_t2,temp_patch_t2),axis=0)
            label_t2 = np.concatenate((label_t2,label_t2),axis=0)

    patch_t1 = np.reshape(patch_t1, [-1, ksize*stride, 3])
    patch_t2 = np.reshape(patch_t2, [-1, ksize*stride, 3])

    mean_t1 = np.mean(patch_t1, axis=1)
    mean_t2 = np.mean(patch_t2, axis=1)
    std_t1 = np.std(patch_t1, axis=1)
    std_t2 = np.std(patch_t2, axis=1)

    feature_t1 = np.concatenate((mean_t1, std_t1), axis=1)
    feature_t2 = np.concatenate((mean_t2, std_t2), axis=1)

    return feature_t1, feature_t2, label_t1, label_t2

def kmeans_hist(trn_feature, val_feature, tst_feature, dim=1000):
    print('')
    logging.info('Kmeans clustering started......')
    num = 100000
    perm = np.random.permutation(trn_feature.shape[0])
    estimator = KMeans(n_clusters=dim,verbose=0, max_iter=500,)
    logging.info('fitting on training data.....')

    estimator.fit(trn_feature[perm[0:num],:])
    logging.info('predicting on training data.....')
    trn_label = estimator.labels_
    trn_label = trn_label - trn_label.min()
    logging.info('predicting on validation data.....')
    val_label = estimator.predict(val_feature)
    val_label = val_label-val_label.min()
    logging.info('predicting on testing data.....')
    tst_label = estimator.predict(tst_feature)
    tst_label = tst_label-tst_label.min()
    logging.info('Kmeans clustering done......')

    return trn_label, val_label, tst_label

def ksvm_classify(trn_data, trn_label, val_data, tst_data):

    svc = sklearn.svm.SVC(kernel=kernel_hik, probability=True, verbose=True)

    svc.fit(X=trn_data, y=trn_label)

    pred_trn_prob = svc.predict_log_proba(trn_data)
    pred_val_prob = svc.predict_log_proba(val_data)
    pred_tst_prob = svc.predict_log_proba(tst_data)

    return pred_trn_prob, pred_val_prob, pred_tst_prob

def main(trn_file, val_file, tst_file):

    logging.info('loading training data...')
    trn_feature_t1, trn_feature_t2, trn_label_t1, trn_label_t2 = extract_feature(trn_file)
    logging.info('loading validation data...')
    val_feature_t1, val_feature_t2, val_label_t1, val_label_t2 = extract_feature(val_file)
    logging.info('loading testing data...')
    tst_feature_t1, tst_feature_t2, tst_label_t1, tst_label_t2 = extract_feature(tst_file)

    dim = 1000
    trn_cluster_t1, val_cluster_t1, tst_cluster_t1 = kmeans_hist(trn_feature_t1, val_feature_t1, tst_feature_t1, dim=dim)
    trn_cluster_t2, val_cluster_t2, tst_cluster_t2 = kmeans_hist(trn_feature_t2, val_feature_t2, tst_feature_t2, dim=dim)

    trn_cluster_t1 = np.reshape(trn_cluster_t1, newshape=[trn_label_t1.shape[0],-1])
    val_cluster_t1 = np.reshape(val_cluster_t1, newshape=[val_label_t1.shape[0],-1])
    tst_cluster_t1 = np.reshape(tst_cluster_t1, newshape=[tst_label_t1.shape[0],-1])
    trn_cluster_t2 = np.reshape(trn_cluster_t2, newshape=[trn_label_t2.shape[0],-1])
    val_cluster_t2 = np.reshape(val_cluster_t2, newshape=[val_label_t2.shape[0],-1])
    tst_cluster_t2 = np.reshape(tst_cluster_t2, newshape=[tst_label_t2.shape[0],-1])

    trn_hist_t1 = to_histogram(data=trn_cluster_t1, dim=dim)
    val_hist_t1 = to_histogram(data=val_cluster_t1, dim=dim)
    tst_hist_t1 = to_histogram(data=tst_cluster_t1, dim=dim)
    trn_hist_t2 = to_histogram(data=trn_cluster_t2, dim=dim)
    val_hist_t2 = to_histogram(data=val_cluster_t2, dim=dim)
    tst_hist_t2 = to_histogram(data=tst_cluster_t2, dim=dim)

    trn_prob_t1, val_prob_t1, tst_prob_t1 = ksvm_classify(trn_hist_t1, trn_label_t1, val_hist_t1, tst_hist_t1)
    trn_prob_t2, val_prob_t2, tst_prob_t2 = ksvm_classify(trn_hist_t2, trn_label_t2, val_hist_t2, tst_hist_t2)

    import scipy.io as sio
    mdict = {
        'trn_prob_t1': trn_prob_t1,
        'val_prob_t1': val_prob_t1,
        'tst_prob_t1': tst_prob_t1,
        'trn_prob_t2': trn_prob_t2,
        'val_prob_t2': val_prob_t2,
        'tst_prob_t2': tst_prob_t2,
        'trn_label_t1': trn_label_t1,
        'val_label_t1': val_label_t1,
        'tst_label_t1': tst_label_t1,
        'trn_label_t2': trn_label_t2,
        'val_label_t2': val_label_t2,
        'tst_label_t2': tst_label_t2,
    }
    sio.savemat('result.mat',mdict=mdict)

    return True

if __name__ == '__main__':

    trn_list = os.listdir(trn_dir)
    trn_file = [trn_dir+npz for npz in trn_list]
    #logging.info(trn_file)

    val_list = os.listdir(val_dir)
    val_file = [val_dir+npz for npz in val_list]
    #logging.info(val_file)

    tst_list = os.listdir(tst_dir)
    tst_file = [tst_dir+npz for npz in tst_list]
    #logging.info(tst_file)

    main(trn_file, val_file, tst_file)
