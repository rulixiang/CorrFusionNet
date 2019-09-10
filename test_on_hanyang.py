# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 19:51:35 2019

@author: rulix
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from matplotlib import image as mpimg
from PIL import Image
Image.MAX_IMAGE_PIXELS = 20000*20000

logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', help='GPU ID', default='2')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = './model-vgg16/'
image_t1_path = './p1.jpg'
image_t2_path = './p2.jpg'

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

saver = tf.train.import_meta_graph(model_path+'cnn_vgg16_epoches=20_batchsize=32.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint(model_path))

image_t1 = Image.open(image_t1_path)#.astype(np.float32)
image_t1 = np.array(image_t1).astype(np.float32)
image_t2 = Image.open(image_t2_path)#.astype(np.float32)
image_t2 = np.array(image_t2).astype(np.float32)

image_t1 = image_t1 / np.max(image_t1) - 0.5
image_t2 = image_t2 / np.max(image_t2) - 0.5

img_shape = np.shape(image_t1)
pat_col, pat_row = 200, 200
step = 50

row = int(img_shape[0]/step-pat_row/step+1)
col = int(img_shape[1]/step-pat_col/step+1)

print(col)
class_t1 = np.zeros(shape=(row,col))
img_batch_t1 = np.zeros((col, pat_col,pat_row,3)).astype(np.float32)
class_t2 = np.zeros(shape=(row,col))
img_batch_t2 = np.zeros((col, pat_col,pat_row,3)).astype(np.float32)

graph = tf.get_default_graph()
inputs_t1 = graph.get_tensor_by_name(name='inputs_t1:0')
inputs_t2 = graph.get_tensor_by_name(name='inputs_t2:0')
label_t1 = graph.get_tensor_by_name(name='ArgMax:0')
label_t2 = graph.get_tensor_by_name(name='ArgMax_1:0')


for k1 in range(row):
        #cnt=0
        for k2 in range(col):
                #print(k2)
                img_batch_t1[k2,:] = image_t1[k1*step:k1*step+200,k2*step:k2*step+200,:]
                img_batch_t2[k2,:] = image_t2[k1*step:k1*step+200,k2*step:k2*step+200,:]

        t1 = np.zeros(col)
        t2 = np.zeros(col)
        for k2 in range(0, col, 64):
                ub = int(np.min((k2+64, col)))
                feed_dict = {inputs_t1: img_batch_t1[k2:ub,:], inputs_t2: img_batch_t1[k2:ub,:]}
                tmp_t1, tmp_t2 = sess.run([label_t1, label_t2], feed_dict=feed_dict)
                t1[k2:ub] = tmp_t1
                t2[k2:ub] = tmp_t2

        class_t1[k1,:] = t1
        class_t2[k1,:] = t2

        logging.info('Row: %2d......'%(k1))


colormap = np.array([[128,   0,   0],#公共用地
                     [128, 128,   0],#商业区
                     [  0,   0, 128],#水
                     [  0, 128,   0],#农田
                     [  0, 192,   0],#绿地
                     [128, 128, 128],#公共交通
                     [192, 128,   0],#工业区
                     [ 64,   0, 128],#住宅区1
                     [128,   0, 128],#住宅区2
                     [192,   0, 128],#住宅区3
                     [128,  64,   0],#道路
                     [192, 128, 128],#停车场
                     [ 64,  64,   0],#裸地
                     [  0,  64, 128], ], dtype=np.float32)/255#操场

image_png_t1 = np.zeros(shape=(row,col,3))
image_png_t2 = np.zeros(shape=(row,col,3))

for k1 in range(row):
        for k2 in range(col):
                image_png_t1[k1,k2,:]=colormap[int(class_t1[k1,k2]),:]
                image_png_t2[k1,k2,:]=colormap[int(class_t2[k1,k2]),:]

image_png_t1 = np.repeat(image_png_t1,repeats=step,axis=0)
image_png_t1 = np.repeat(image_png_t1,repeats=step,axis=1)
image_png_t2 = np.repeat(image_png_t2,repeats=step,axis=0)
image_png_t2 = np.repeat(image_png_t2,repeats=step,axis=1)

plt.imsave('image_t1.png',image_png_t1)
plt.imsave('image_t2.png',image_png_t2)

import scipy.io as sio
sio.savemat('result.mat',mdict={'img1':class_t1,'img2':class_t2})

sess.close()
