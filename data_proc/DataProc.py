# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:07:46 2019

@author: rulix
"""

import os
import re
import shutil
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
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
    cm = metrics.confusion_matrix(label_t1, label_t2)
    np.set_printoptions(precision=3, suppress=True)
    #print(cm)
    num = np.sum(label_t1!=label_t2)

    filename_t1 = ['']*len(idx_col)
    filename_t2 = ['']*len(idx_col)
    #cnt=1    
    for k in range(len(idx_col)):
        filename_t1[k] = cfg.dir_t1+cfg.class_name[label_t1[k]]+'/'+str(idx_col[k]+1)+'_'+str(idx_row[k]+1)+'.png'
        filename_t2[k] = cfg.dir_t2+cfg.class_name[label_t2[k]]+'/'+str(idx_col[k]+1)+'_'+str(idx_row[k]+1)+'.png'
        '''
        if label_t1[k]!=label_t2[k]:
            img_t1 = mpimg.imread(filename_t1[k])
            img_t2 = mpimg.imread(filename_t2[k])
            
            plt.close()
            plt.subplot('121')
            plt.imshow(img_t1)
            plt.title(cfg.class_name[label_t1[k]])
            plt.subplot('122')
            plt.imshow(img_t2)
            plt.title(cfg.class_name[label_t2[k]])
            plt.show()
            
            a=input('input right class：')
            
            if a is '1':
                filename = cfg.dir_t2+cfg.class_name[label_t2[k]]+'/'+str(idx_col[k]+1)+'_'+str(idx_row[k]+1)+'.png'
                new_filename = cfg.dir_t2+cfg.class_name[label_t1[k]]+'/'+str(idx_col[k]+1)+'_'+str(idx_row[k]+1)+'.png'
                print('Moving %s to %s\n'%(filename, new_filename))
                shutil.move(filename, new_filename)
            elif a is '2':
                filename = cfg.dir_t1+cfg.class_name[label_t1[k]]+'/'+str(idx_col[k]+1)+'_'+str(idx_row[k]+1)+'.png'
                new_filename = cfg.dir_t1+cfg.class_name[label_t2[k]]+'/'+str(idx_col[k]+1)+'_'+str(idx_row[k]+1)+'.png'
                print('Moving %s to %s\n'%(filename, new_filename))
                shutil.move(filename, new_filename)
            else:
                print('No operation between %s and %s\n'%(filename_t1[k],filename_t2[k]))
            
            #plt.show()
            print('%3d / %3d\n\n'%(cnt, num))
            cnt = cnt+1
        '''

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
    
    new_label_map_t1 = np.zeros_like(labelmap_t1)
    new_label_map_t2 = np.zeros_like(labelmap_t2)
                
    new_color_map_t1, new_label_map_t1 = genNolap(colormap_t1, labelmap_t1, cfg=cfg)
    new_color_map_t2, new_label_map_t2 = genNolap(colormap_t2, labelmap_t2, cfg=cfg)
    
    for row in range(np.shape(new_label_map_t1)[0]):
        for col in range(np.shape(new_label_map_t1)[1]):
            lt1 = int(new_label_map_t1[row,col])
            lt2 = int(new_label_map_t2[row,col])
            if lt1==0 or lt2==0:
                new_label_map_t1[row,col] = 0
                new_label_map_t2[row,col] = 0
                new_color_map_t1[row,col,:] = 0
                new_color_map_t2[row,col,:] = 0

                #a = plt.imread(filename_t1)
                #plt.imsave(filename_t2, a)
    
    new_color_map_t1 = np.repeat(new_color_map_t1, 10, axis=0)
    new_color_map_t2 = np.repeat(new_color_map_t2, 10, axis=0)
    new_color_map_t1 = np.repeat(new_color_map_t1, 10, axis=1)
    new_color_map_t2 = np.repeat(new_color_map_t2, 10, axis=1)

    print(np.sum(new_label_map_t1!=0))
    print(np.sum(new_label_map_t1!=new_label_map_t2))
    
    plt.imsave('co_mask_t1.png',new_color_map_t1)
    plt.imsave('co_mask_t2.png',new_color_map_t2)
    
    #os.mkdir('2014')
    #os.mkdir('2014/semisure')
    #os.mkdir('2016')
    #os.mkdir('2016/semisure')
    '''
    for ndir in cfg.class_name:
        os.mkdir('2014/semisure/'+ndir)
        os.mkdir('2016/semisure/'+ndir)
    import shutil
    for row in range(np.shape(new_label_map_t1)[0]):
        for col in range(np.shape(new_label_map_t1)[1]):
            lt1 = int(new_label_map_t1[row,col])
            lt2 = int(new_label_map_t2[row,col])
            if lt1!=0 and lt2==0:
                filename_t1 = '2016/notsure/'+str(col+1)+'_'+str(row+1)+'.png'
                filename_t2 = '2016/semisure/'+cfg.class_name[lt1]+'/'+str(col+1)+'_'+str(row+1)+'.png'
                shutil.move(filename_t1,filename_t2)
                #a = plt.imread(filename_t1)
                #plt.imsave(filename_t2, a)

            elif lt1==0 and lt2!=0:
                filename_t1 = '2014/semisure/'+cfg.class_name[lt2]+'/'+str(col+1)+'_'+str(row+1)+'.png'
                filename_t2 = '2014/notsure/'+str(col+1)+'_'+str(row+1)+'.png'
                shutil.move(filename_t2,filename_t1)
                #a = plt.imread(filename_t2)
                #plt.imsave(filename_t1, a)
    '''

    filename_t1, filename_t2, label_t1, label_t2 = genFileList(new_label_map_t1, new_label_map_t2, cfg=cfg)

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

    #saveNpy(out_dir='./data_small/trn/',file=file_dict, index=trn_idx)
    #saveNpy(out_dir='./data_small/val/',file=file_dict, index=val_idx)
    #saveNpy(out_dir='./data_small/tst/',file=file_dict, index=tst_idx)

    return True

if __name__ == "__main__":
    cfg = config()

    main(cfg=cfg)
