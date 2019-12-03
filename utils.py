import time

import h5py
import numpy as np
import sklearn.metrics as metrics


def DecodeH5(h5file=None):
    
    file = h5py.File(name=h5file, mode='r')
    
    data = file['image'].value.astype(np.float32)/255-0.5
    label = file['label'].value.astype(np.int8)-1
    return data, label
def LoadNpy(filename=None):

    npy = np.load(file=filename)
    image_t1 = npy['image_t1']
    image_t1 = image_t1.astype(np.float32)/np.max(image_t1)-0.5
    image_t2 = npy['image_t2'] 
    image_t2 = image_t2.astype(np.float32)/np.max(image_t2)-0.5
    label_t1 = npy['label_t1'] - 1
    label_t2 = npy['label_t2'] - 1

    
    ## shuffle
    #perm_index = np.random.permutation(len(label_t1))
    #label_t1 = label_t1[perm_index]
    #image_t1 = image_t1[perm_index,:]
    
    return image_t1, image_t2, label_t1, label_t2
    #return np.concatenate((image_t1, image_t2),axis=0), np.concatenate((image_t2, image_t1),axis=0), np.concatenate((label_t1, label_t2),axis=0), np.concatenate((label_t2, label_t1),axis=0)

def Accuracy(pred_t1, pred_t2, label_t1, label_t2):
    oa_t1 = metrics.accuracy_score(y_true=label_t1, y_pred=pred_t1)
    oa_t2 = metrics.accuracy_score(y_true=label_t2, y_pred=pred_t2)

    pred_bi = np.equal(pred_t1, pred_t2).astype(np.int16)
    label_bi = np.equal(label_t1, label_t2).astype(np.int16)
    oa_bi = metrics.accuracy_score(y_true=label_bi, y_pred=pred_bi,)
    #oa_bi = metrics.precision_score(y_true=label_bi,y_pred=pred_bi)
    '''
    cnt = 0.
    for k1 in range(len(pred_t1)):
        if pred_t1[k1]==label_t1[k1] and pred_t2[k1]==label_t2[k1]:
            cnt = cnt + 1 
    oa_tr = cnt/float(len(pred_t1))
    '''
    oa_tr = np.sum((pred_t1==label_t1)&((pred_t2==label_t2)))/float(len(pred_t1))
    #print('ok')

    return oa_t1, oa_t2, oa_bi, oa_tr
