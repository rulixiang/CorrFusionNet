# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:05:57 2019

@author: rulix
"""
import os
import logging

import numpy as np
import scipy.io as sio

trn_dir = '../data_small/trn/'
val_dir = '../data_small/val/'
tst_dir = '../data_small/tst/'


def LoadNpy(filename=None):

    npy = np.load(file=filename)
    image_t1 = npy['image_t1']
    image_t1 = image_t1.astype(np.float32)/np.max(image_t1)#-0.5
    image_t2 = npy['image_t2'] 
    image_t2 = image_t2.astype(np.float32)/np.max(image_t2)#-0.5
    label_t1 = npy['label_t1'] - 1
    label_t2 = npy['label_t2'] - 1

    return image_t1, image_t2, label_t1, label_t2


def extract_label(file_list):

    label_t1 = None
    label_t2 = None

    for file in file_list:

        image_t1, image_t2, temp_label_t1, temp_label_t2 = LoadNpy(file)

        if label_t1 is None:
            label_t1 = temp_label_t1
            label_t2 = temp_label_t2
        else:
            label_t1 = np.concatenate((label_t1,temp_label_t1),axis=0)
            label_t2 = np.concatenate((label_t2,temp_label_t2),axis=0)

    return label_t1, label_t2



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

    trn_label_t1, trn_label_t2 = extract_label(trn_file)
    tst_label_t1, tst_label_t2 = extract_label(tst_file)
    val_label_t1, val_label_t2 = extract_label(val_file)

    sio.savemat('label.mat',mdict={'trn_label_t1':trn_label_t1,'trn_label_t2':trn_label_t2,'tst_label_t1':tst_label_t1,'tst_label_t2':tst_label_t2,'val_label_t1':val_label_t1,'val_label_t2':val_label_t2})
