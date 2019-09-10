import h5py
import time
import numpy as np

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

    return image_t1, image_t2, label_t1, label_t2

def SaveModel():
    
    return True