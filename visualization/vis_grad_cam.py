import argparse
import os
import tensorflow as tf
import numpy as np
from scipy import misc
from matplotlib import pyplot as plt

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_file', help='output file', default='./mask_img.png')
    parser.add_argument('-i', '--in_file', help='input file', default='./5_68.png')
    parser.add_argument('-m', '--mod_dir', help='model dir', default='./model_test')
    parser.add_argument('-v', '--view', help='view', type=float, default=1)
    parser.add_argument('-c', '--category', help='category id', type=float, default=11)
    parser.add_argument('-g', '--gpu', help='gpu device id', default='0')
    args = parser.parse_args()
    return args

def main(args=None, inputs_name='inputs_t1:0',probs_name='probs_t1:0'):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.import_meta_graph(args.mod_dir+'/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(args.mod_dir))

    graph = tf.get_default_graph()
    image_inputs = graph.get_tensor_by_name(name='inputs_t1:0')
    label_inputs = graph.get_tensor_by_name(name='label_t1:0')
    probs = graph.get_tensor_by_name(name='probs_t1:0')
    loss = graph.get_tensor_by_name(name='softmax_cross_entropy_loss/value:0')
    #conv_layer = graph.get_tensor_by_name(name='conv_layers/block5_pool/MaxPool:0')
    conv_layer = graph.get_tensor_by_name(name='conv_layers/block5_conv3/Relu:0')

    img = misc.imread(args.in_file).astype(np.float32)
    img = img-img.min()
    img = img/img.max() - 0.5
    img = np.expand_dims(img, axis=0)
    label = np.expand_dims(args.category, axis=0).astype(np.uint8)

    grads = tf.gradients(loss, conv_layer)[0]

    norm_grads = tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={image_inputs: img, label_inputs: label})
    output = output[0]           # [10,10,2048]
    grads_val = grads_val[0]	 # [10,10,2048]
    weights = np.mean(grads_val, axis = (0, 1)) 			# [2048]
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)	# [10,10]
    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
        # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    img = img[0,:]
    img = img - img.min()
    img = img / img.max()
    cam = misc.imresize(cam, [img.shape[0],img.shape[1]]).astype(np.float32)
    cam = cam / np.max(cam)
    
    cmap = plt.get_cmap('seismic')
    cam = cmap(cam)

    img = 0.4*img + 0.6*cam[:,:,0:3]

    plt.imsave('cam.png', cam)
    plt.imsave('cam_mask.png', img)

    sess.close()

    return True

if __name__=='__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    main(args=args)
