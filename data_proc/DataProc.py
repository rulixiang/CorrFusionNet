# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:07:46 2019

@author: rulix
"""

import os
import re

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from config import config

def genMap(file_dir, shape=None, cm=None):

    color_map = np.zeros(np.append(shape,3))
    label_map = np.zeros(shape=shape, dtype=np.int16)

    dir_tmp = os.listdir(file_dir)

    for d in dir_tmp:
        label = float(d.split('-')[0])
        img_list = os.listdir(file_dir+d)

        for img in img_list:
            loc_x = int(re.split(r'[_.]', img)[0])-1
            loc_y = int(re.split(r'[_.]', img)[1])-1
            label_map[loc_y, loc_x] = label
            color_map[loc_y, loc_x,:]=cm[int(label-1),:]

    return color_map, label_map

def genNolap(color_map, label_map, cfg=None):

    step = cfg.step
    new_shape = np.floor(np.array(label_map.shape)/step).astype(np.int16)
    new_label_map = np.zeros(new_shape)
    new_color_map = np.zeros(np.append(new_shape,3))

    for k1 in range(new_shape[0]):
        for k2 in range(new_shape[1]):
            if label_map[step*k1,step*k2]!=0:
                new_label_map[k1,k2] = label_map[step*k1,step*k2]
                new_color_map[k1,k2,:] = color_map[step*k1,step*k2,:]

    return new_color_map, new_label_map

def genFileList(label_map_t1, label_map_t2, cfg=None):

    step = cfg.step
    index = np.where((label_map_t1!=0)&(label_map_t2!=0))
    idx_row = index[0]
    idx_col = index[1]
    label_t1 = label_map_t1[idx_row, idx_col].astype(np.int16)
    label_t2 = label_map_t2[idx_row, idx_col].astype(np.int16)
    filename_t1 = ['']*len(idx_col)
    filename_t2 = ['']*len(idx_col)
    for k in range(len(idx_col)):
        filename_t1[k] = cfg.dir_t1+cfg.class_name[label_t1[k]]+'/'+str(idx_col[k]+1)+'_'+str(idx_row[k]+1)+'.png'
        filename_t2[k] = cfg.dir_t2+cfg.class_name[label_t2[k]]+'/'+str(idx_col[k]+1)+'_'+str(idx_row[k]+1)+'.png'


    return filename_t1, filename_t2, label_t1, label_t2

def saveNpy(out_dir=None, file=None, index=None):
    if os.path.exists(out_dir) is False:
        os.mkdir(out_dir)
    step = 5000
    filename_t1 = file['filename_t1']
    filename_t2 = file['filename_t2']
    label_t1 = file['label_t1']
    label_t2 = file['label_t2']

    for k in range(0, len(index), step):
        ub = min([len(index), k+step])
        temp_idx = index[k:ub]
        size = len(temp_idx)

        image_t1 = np.zeros((size,200,200,3),dtype=np.uint8)
        image_t2 = np.zeros((size,200,200,3),dtype=np.uint8)

        for k1 in range(len(temp_idx)):
            idx = int(temp_idx[k1])
            fname_t1 = filename_t1[idx]
            fname_t2 = filename_t2[idx]
            ##
            no = fname_t1.split('/')[-2]
            no = no.split('-')[0]
            assert float(no)==label_t1[temp_idx[k1]]

            image_t1[k1,:] = mpimg.imread(fname_t1)*255
            image_t2[k1,:] = mpimg.imread(fname_t2)*255
            if k1%100==0:
                print(k1)
        out_file = out_dir+str(k)+'-'+str(ub)+'.npz'
        np.savez(out_file,image_t1=image_t1, image_t2=image_t2, label_t1=label_t1[temp_idx], label_t2=label_t2[temp_idx])

    return True

def main(cfg=None):
    colormap_t1, labelmap_t1 = genMap(cfg.dir_t1, cfg.shape, cfg.colormap)
    colormap_t2, labelmap_t2 = genMap(cfg.dir_t2, cfg.shape, cfg.colormap)
    #plt.imshow(colormap_t1)
    #plt.show()
    
    new_labelmap_t1 = np.zeros_like(labelmap_t1)
    new_labelmap_t2 = np.zeros_like(labelmap_t2)
                
    new_color_map_t1, new_labelmap_t1 = genNolap(colormap_t1, labelmap_t1, cfg=cfg)
    new_color_map_t2, new_labelmap_t2 = genNolap(colormap_t2, labelmap_t2, cfg=cfg)

    plt.imsave('label_t1.png',new_color_map_t1)
    plt.imsave('label_t2.png',new_color_map_t2)


    filename_t1, filename_t2, label_t1, label_t2 = genFileList(new_labelmap_t1, new_labelmap_t2, cfg=cfg)

    trn_rate = 0.7
    tst_rate = 0.2
    val_rate = 0.1

    trn_num = np.floor(trn_rate*len(filename_t1)).astype(np.int32)
    val_num = np.floor(val_rate*len(filename_t1)).astype(np.int32)

    perm = np.random.permutation(len(filename_t1))

    trn_idx = perm[0:trn_num]
    val_idx = perm[trn_num:trn_num+val_num]
    tst_idx = perm[trn_num+val_num:]

    file_dict = {
        'filename_t1': filename_t1,
        'filename_t2': filename_t2,
        'label_t1': label_t1,
        'label_t2': label_t2
    }

    saveNpy(out_dir='./data_small/trn/',file=file_dict, index=trn_idx)
    saveNpy(out_dir='./data_small/val/',file=file_dict, index=val_idx)
    saveNpy(out_dir='./data_small/tst/',file=file_dict, index=tst_idx)

    return True

if __name__ == "__main__":
    cfg = config()
    main(cfg=cfg)
