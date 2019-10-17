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
Image.MAX_IMAGE_PIXELS = 50000 * 50000

logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', help='GPU ID', default='-1')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = './model-vgg16/'
image_t1_path = './wuhan2014.jpg'
image_t2_path = './wuhan2016.jpg'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

saver = tf.train.import_meta_graph(model_path + 'cnn_vgg16_epoches=20_batchsize=32.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint(model_path))


## predicting time 1 image
def predict(image_path=None,saver=None,step=200,inputs_name='inputs_t1:0',probs_name='probs_t1:0'):
    image_t = Image.open(image_path)  #.astype(np.float32)
    image_t = np.array(image_t).astype(np.float32)
    image_t = image_t / np.max(image_t) - 0.5

    img_shape = np.shape(image_t)
    pat_col, pat_row = 200, 200
    row = int(img_shape[0] / step - pat_row / step + 1)
    col = int(img_shape[1] / step - pat_col / step + 1)

    print(col)
    class_num = 14
    prob_t = np.zeros(shape=(row, col, class_num))
    img_batch = np.zeros((col, pat_col, pat_row, 3)).astype(np.float32)

    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name(name=inputs_name)
    probs = graph.get_tensor_by_name(name=probs_name)

    for k1 in range(row):
        for k2 in range(col):
            #print(k2)
            img_batch[k2, :] = image_t[k1*step: k1*step+200, k2*step: k2*step+200, :]

        tmp = np.zeros((col, class_num))

        for k2 in range(0, col, 64):
            ub = int(np.min((k2 + 64, col)))
            feed_dict = {inputs: img_batch[k2:ub, :]}
            tmp_t1 = sess.run(probs, feed_dict=feed_dict)
            tmp[k2:ub, :] = tmp_t1

        prob_t[k1, :] = tmp
        logging.info('Row: %2d......' % (k1))

    return prob_t


prob_t1 = predict(image_path=image_t1_path,saver=saver,inputs_name='inputs_t1:0',probs_name='probs_t1:0')
prob_t2 = predict(image_path=image_t2_path,saver=saver,inputs_name='inputs_t2:0',probs_name='probs_t2:0')

sess.close()

import scipy.io as sio
sio.savemat('cnn_vgg16_test_all.mat',mdict={'prob_t1': prob_t1,'prob_t2': prob_t2})
'''
colormap = np.array(
    [
        [128, 0, 0],  #公共用地
        [128, 128, 0],  #商业区
        [0, 0, 128],  #水
        [0, 128, 0],  #农田
        [0, 192, 0],  #绿地
        [128, 128, 0],  #公共交通
        [192, 128, 0],  #工业区
        [64, 0, 128],  #住宅区1
        [128, 0, 128],  #住宅区2
        [192, 0, 128],  #住宅区3
        [128, 64, 0],  #道路
        [192, 128, 128],  #停车场
        [64, 64, 0],  #裸地
        [0, 64, 128],
    ],
    dtype=np.float32) / 255  #操场

image_png_t1 = np.zeros(shape=(row, col, 3))
image_png_t2 = np.zeros(shape=(row, col, 3))

for k1 in range(row):
    for k2 in range(col):
        image_png_t1[k1, k2, :] = colormap[int(class_t1[k1, k2]), :]
        image_png_t2[k1, k2, :] = colormap[int(class_t2[k1, k2]), :]

image_png_t1 = np.repeat(image_png_t1, repeats=step, axis=0)
image_png_t1 = np.repeat(image_png_t1, repeats=step, axis=1)
image_png_t2 = np.repeat(image_png_t2, repeats=step, axis=0)
image_png_t2 = np.repeat(image_png_t2, repeats=step, axis=1)

plt.imsave('image_t1.png', image_png_t1)
plt.imsave('image_t2.png', image_png_t2)
'''
