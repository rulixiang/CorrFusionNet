import argparse
import logging
import os
import time

import h5py
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sklearn.metrics as metrics
import tensorflow as tf
from sklearn.manifold import TSNE

import utils




parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', help='gpu device ID', default='0')
parser.add_argument('-m', '--model_dir', help='model directory', default='./cnn-sdcca-res/model/0/')
parser.add_argument('-o', '--output', help='output file', default='./visualization/test_embeddings.mat')
parser.add_argument('-tst', '--tst_dir', help='testing file dir', default='./data_small/tst/')
parser.add_argument('-trn', '--trn_dir', help='training file dir', default='./data_small/trn/')
parser.add_argument('-val', '--val_dir', help='validation file dir', default='./data_small/val/')
args = parser.parse_args()


def vis_embeddings(embeddings=None, labels=None, save_name=None):
    
    plt.figure()
    plt.scatter(embeddings[:,1], embeddings[:,2], c=labels, label=labels, cmap='hsv')
    plt.savefig('./visualization/'+save_name)
    return True

def test(sess, graph, file_list=None):

    inputs_t1 = graph.get_tensor_by_name(name='inputs_t1:0')
    inputs_t2 = graph.get_tensor_by_name(name='inputs_t2:0')

    #embeddings_t1 = graph.get_tensor_by_name(name='losses/softmax/MatMul:0')
    embeddings_t1 = graph.get_tensor_by_name(name='dense3_t1/BiasAdd:0')
    #embeddings_t2 = graph.get_tensor_by_name(name='losses/softmax_1/MatMul:0')
    embeddings_t2 = graph.get_tensor_by_name(name='dense3_t2/BiasAdd:0')

    embeddings_label_t1 = None
    embeddings_label_t2 = None
    label_t1 = None
    label_t2 = None

    for pfile in file_list:

        print('evaluating on file: %s......'%(pfile))
        xbatch1, xbatch2, ybatch1, ybatch2 = utils.LoadNpy(pfile)

        if label_t1 is None:
            label_t1 = ybatch1
            label_t2 = ybatch2
        else:
            label_t1 = np.concatenate((label_t1, ybatch1), axis=0)
            label_t2 = np.concatenate((label_t2, ybatch2), axis=0)

        for k1 in range(0, np.shape(xbatch1)[0], 32):
            lb = int(k1)
            ub = int(np.min((lb+32,np.shape(xbatch1)[0])))
            #sess.run(base_model.local_init)
            feed_dict = {inputs_t1: xbatch1[lb:ub, :], inputs_t2: xbatch2[lb:ub, :]}
            tmp_embeddings_t1, tmp_embeddings_t2 = sess.run([embeddings_t1, embeddings_t2], feed_dict=feed_dict)

            if embeddings_label_t1 is None:
                embeddings_label_t1 = tmp_embeddings_t1
                embeddings_label_t2 = tmp_embeddings_t2
            else:
                embeddings_label_t1 = np.concatenate((embeddings_label_t1, tmp_embeddings_t1), axis=0)
                embeddings_label_t2 = np.concatenate((embeddings_label_t2, tmp_embeddings_t2), axis=0)

    #embeddings_label_t1 = np.expand_dims(embeddings_label_t1, axis=1)
    #embeddings_label_t2 = np.expand_dims(embeddings_label_t2, axis=1)

    return label_t1, label_t2, embeddings_label_t1, embeddings_label_t2

def main(args=None, trn_file=None, val_file=None, tst_file=None):

    dir_list = os.listdir(args.model_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    model_name = args.model_dir + 'model.ckpt.meta'
    saver = tf.train.import_meta_graph(model_name)
    saver.restore(sess, tf.train.latest_checkpoint(args.model_dir))

    graph = tf.get_default_graph()
        
    trn_label_t1, trn_label_t2, trn_embeddings_t1, trn_embeddings_t2 = test(sess=sess, graph=graph, file_list=trn_file)
    val_label_t1, val_label_t2, val_embeddings_t1, val_embeddings_t2 = test(sess=sess, graph=graph, file_list=val_file)
    tst_label_t1, tst_label_t2, tst_embeddings_t1, tst_embeddings_t2 = test(sess=sess, graph=graph, file_list=tst_file)

    sess.close()

    vis_embeddings(embeddings=trn_embeddings_t1, labels=trn_label_t1, save_name='trn_t1.png')
    vis_embeddings(embeddings=trn_embeddings_t2, labels=trn_label_t2, save_name='trn_t2.png')
    vis_embeddings(embeddings=val_embeddings_t1, labels=val_label_t1, save_name='val_t1.png')
    vis_embeddings(embeddings=val_embeddings_t2, labels=val_label_t2, save_name='val_t2.png')
    vis_embeddings(embeddings=tst_embeddings_t1, labels=tst_label_t1, save_name='tst_t1.png')
    vis_embeddings(embeddings=tst_embeddings_t2, labels=tst_label_t2, save_name='tst_t2.png')

    perplexity = 10
    n_iter = 2000
    #val_t1_2d = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, verbose=1).fit_transform(val_embeddings_t1)
    #val_t2_2d = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, verbose=1).fit_transform(val_embeddings_t2)
    #val_t1_3d = TSNE(n_components=3).fit_transform(val_embeddings_t1)
    #val_t2_3d = TSNE(n_components=3).fit_transform(val_embeddings_t2)

    #tst_t1_2d = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, verbose=1).fit_transform(tst_embeddings_t1)
    #tst_t2_2d = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, verbose=1).fit_transform(tst_embeddings_t2)
    #tst_t1_3d = TSNE(n_components=3).fit_transform(tst_embeddings_t1)
    #tst_t2_3d = TSNE(n_components=3).fit_transform(tst_embeddings_t2)

    mdict = {
        'val_embedding_t1': val_embeddings_t1,
        'val_embedding_t2': val_embeddings_t2,
        'trn_embedding_t1': trn_embeddings_t1,
        'trn_embedding_t2': trn_embeddings_t2,
        'tst_embedding_t1': tst_embeddings_t1,
        'tst_embedding_t2': tst_embeddings_t2,
        'val_label_t1': val_label_t1,
        'val_label_t2': val_label_t2,
        'tst_label_t1': tst_label_t1,
        'tst_label_t2': tst_label_t2,
        'trn_label_t1': trn_label_t1,
        'trn_label_t2': trn_label_t2
    }

    sio.savemat(args.output, mdict=mdict)

    return None

if __name__=='__main__':

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

    main(args, trn_file, val_file, tst_file)
