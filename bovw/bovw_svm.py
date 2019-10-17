# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:05:57 2019

@author: rulix
"""
import os
import logging
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

trn_dir = '../data_small/trn/'
val_dir = '../data_small/val/'
tst_dir = '../data_small/tst/'

def metric(y_true, y_pred):

    kc = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    oa = sklearn.metrics.accuracy_score(y_true, y_pred)

    return oa, kc

def LoadNpy(filename=None):

    npy = np.load(file=filename)
    image_t1 = npy['image_t1']
    image_t1 = image_t1.astype(np.float32)/np.max(image_t1)#-0.5
    image_t2 = npy['image_t2'] 
    image_t2 = image_t2.astype(np.float32)/np.max(image_t2)#-0.5
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

def to_histogram(data, dim=1000):

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

        image_t1, image_t2, temp_label_t1, temp_label_t2 = LoadNpy(file)

        temp_patch_t1 = extract_patch(image=image_t1, ksize=[1,ksize,ksize,1], stride=[1,stride,stride,1])
        temp_patch_t2 = extract_patch(image=image_t2, ksize=[1,ksize,ksize,1], stride=[1,stride,stride,1])

        if patch_t1 is None:
            patch_t1 = temp_patch_t1
            label_t1 = temp_label_t1
            patch_t2 = temp_patch_t2
            label_t2 = temp_label_t2
        else:
            patch_t1 = np.concatenate((patch_t1,temp_patch_t1),axis=0)
            label_t1 = np.concatenate((label_t1,temp_label_t1),axis=0)
            patch_t2 = np.concatenate((patch_t2,temp_patch_t2),axis=0)
            label_t2 = np.concatenate((label_t2,temp_label_t2),axis=0)

    patch_t1 = np.reshape(patch_t1, [-1, ksize*stride, 3])
    patch_t2 = np.reshape(patch_t2, [-1, ksize*stride, 3])

    mean_t1 = np.mean(patch_t1, axis=1)
    mean_t2 = np.mean(patch_t2, axis=1)
    std_t1 = np.std(patch_t1, axis=1)
    std_t2 = np.std(patch_t2, axis=1)

    feature_t1 = np.concatenate((mean_t1, std_t1), axis=1)
    feature_t2 = np.concatenate((mean_t2, std_t2), axis=1)

    return feature_t1, feature_t2, label_t1, label_t2

def kmeans_hist(trn_feature, dim=1000, num=100000):
    print('')
    logging.info('Kmeans clustering started......')

    perm = np.random.permutation(trn_feature.shape[0])
    kmeans_model = KMeans(n_clusters=dim,verbose=0, max_iter=500,)
    logging.info('fitting on training data.....')

    kmeans_model.fit(trn_feature[perm[0:num],:])
    logging.info('predicting on training data.....')
    trn_label = kmeans_model.predict(trn_feature)
    trn_label = trn_label - trn_label.min()
    #logging.info('predicting on validation data.....')
    #val_label = kmeans_model.predict(val_feature)
    #val_label = val_label-val_label.min()
    #logging.info('predicting on testing data.....')
    #tst_label = kmeans_model.predict(tst_feature)
    #tst_label = tst_label-tst_label.min()
    #logging.info('Kmeans clustering done......')

    return kmeans_model, trn_label 

def ksvm_train(trn_data, trn_label):

    svc_model = sklearn.svm.SVC(kernel=kernel_hik, probability=True, verbose=False)

    svc_model.fit(X=trn_data, y=trn_label)

    pred_trn_prob = svc_model.predict_proba(trn_data)
    #pred_val_prob = svc.predict_log_proba(val_data)
    #pred_tst_prob = svc.predict_log_proba(tst_data)

    return svc_model, pred_trn_prob

def test(file_list, svc_model_t1=None, kmeans_model_t1=None, svc_model_t2=None, kmeans_model_t2=None):

    hist_t1 = None
    hist_t2 = None 
    label_t1 = None
    label_t2 = None 

    for file in file_list:

        feature_t1, feature_t2, tmp_label_t1, tmp_label_t2 = extract_feature([file])

        tmp_cluster_t1 = kmeans_model_t1.predict(feature_t1)
        tmp_cluster_t2 = kmeans_model_t2.predict(feature_t2)

        del feature_t2
        del feature_t1

        tmp_cluster_t2 = np.reshape(tmp_cluster_t2, newshape=[tmp_label_t2.shape[0],-1])
        tmp_cluster_t1 = np.reshape(tmp_cluster_t1, newshape=[tmp_label_t1.shape[0],-1])

        tmp_hist_t1 = to_histogram(data=tmp_cluster_t1)
        tmp_hist_t2 = to_histogram(data=tmp_cluster_t2)

        del tmp_cluster_t2
        del tmp_cluster_t1

        if hist_t1 is None:
            hist_t1 = tmp_hist_t1
            label_t1 = tmp_label_t1
            hist_t2 = tmp_hist_t2
            label_t2 = tmp_label_t2
        else:
            hist_t1 = np.concatenate((hist_t1,tmp_hist_t1),axis=0)
            label_t1 = np.concatenate((label_t1,tmp_label_t1),axis=0)
            hist_t2 = np.concatenate((hist_t2,tmp_hist_t2),axis=0)
            label_t2 = np.concatenate((label_t2,tmp_label_t2),axis=0)

    pred_prob_t1 = svc_model_t1.predict_proba(hist_t1)
    pred_t1 = np.argmax(pred_prob_t1, axis=1)
    oa, kc = metric(label_t1, pred_t1)
    logging.info('on time 1, oa is %.4f, kc is %.4f'%(oa, kc))

    pred_prob_t2 = svc_model_t2.predict_proba(hist_t2)
    pred_t2 = np.argmax(pred_prob_t2, axis=1)
    oa, kc = metric(label_t2, pred_t2)
    logging.info('on time 2, oa is %.4f, kc is %.4f'%(oa, kc))

    return pred_prob_t1, pred_prob_t2, hist_t1, hist_t2

def main(trn_file, val_file, tst_file, save_file=None):

    logging.info('loading training data...')
    trn_feature_t1, trn_feature_t2, trn_label_t1, trn_label_t2 = extract_feature(trn_file)
    #logging.info('loading validation data...')
    #val_feature_t1, val_feature_t2, val_label_t1, val_label_t2 = extract_feature(val_file)
    #logging.info('loading testing data...')
    #tst_feature_t1, tst_feature_t2, tst_label_t1, tst_label_t2 = extract_feature(tst_file)

    kmeans_model_t1, trn_cluster_t1 = kmeans_hist(trn_feature_t1)
    kmeans_model_t2, trn_cluster_t2 = kmeans_hist(trn_feature_t2)

    trn_cluster_t1 = np.reshape(trn_cluster_t1, newshape=[trn_label_t1.shape[0],-1])
    trn_cluster_t2 = np.reshape(trn_cluster_t2, newshape=[trn_label_t2.shape[0],-1])

    trn_hist_t1 = to_histogram(data=trn_cluster_t1)
    trn_hist_t2 = to_histogram(data=trn_cluster_t2)

    svc_model_t1, trn_prob_t1 = ksvm_train(trn_hist_t1, trn_label_t1)
    svc_model_t2, trn_prob_t2 = ksvm_train(trn_hist_t2, trn_label_t2)

    logging.info('evaluating on validation set...')
    val_prob_t1, val_prob_t2, val_hist_t1, val_hist_t2 = test(val_file, svc_model_t1, kmeans_model_t1, svc_model_t2, kmeans_model_t2)

    logging.info('evaluating on testing set...')
    tst_prob_t1, tst_prob_t2, tst_hist_t1, tst_hist_t2 = test(tst_file, svc_model_t1, kmeans_model_t1, svc_model_t2, kmeans_model_t2)

    import scipy.io as sio

    mdict = {
        'trn_prob_t1': trn_prob_t1,
        'val_prob_t1': val_prob_t1,
        'tst_prob_t1': tst_prob_t1,
        'trn_prob_t2': trn_prob_t2,
        'val_prob_t2': val_prob_t2,
        'tst_prob_t2': tst_prob_t2,
        'trn_hist_t1': trn_hist_t1,
        'val_hist_t1': val_hist_t1,
        'tst_hist_t1': tst_hist_t1,
        'trn_hist_t2': trn_hist_t2,
        'val_hist_t2': val_hist_t2,
        'tst_hist_t2': tst_hist_t2,
    }

    sio.savemat(save_file,mdict=mdict)
    print('\n\n')
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
    for k in range(10):
        save_file = './results/res_bovw_ksvm_'+str(k)+'.mat'
        print(save_file)
        main(trn_file, val_file, tst_file, save_file)
