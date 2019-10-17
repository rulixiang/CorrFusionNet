# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:02:04 2019

@author: rulix
"""
import sklearn
def metric(label=None, pred=None):
    
    oa = sklearn.metrics.accuracy_score(y_true=label, y_pred=pred)
    kc = sklearn.metrics.cohen_kappa_score(label,pred)
    cm = sklearn.metrics.confusion_matrix(label, pred).astype(np.float)
    _, num = np.unique(label, return_counts=True)
    print(np.shape(num))
    #cm = cm/num
    for k in range(cm.shape[1]):
        cm[:,k]=(cm[:,k])/float(num[k])
    
    return oa, kc, cm

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sb

mat = sio.loadmat('cnn_result_50.mat')
pred_t1 = np.squeeze(mat['pred_label_t1'])
pred_t2 = np.squeeze(mat['pred_label_t2'])
label_t1 = np.squeeze(mat['test_label_t1'])
label_t2 = np.squeeze(mat['test_label_t2'])
'''
pred_t1 = mat['tst_prob_t1']
pred_t1 = np.argmax(pred_t1, axis=1)
label_t1 = mat['tst_label_t1']
label_t1 = np.squeeze(label_t1)

pred_t2 = mat['tst_prob_t2']
pred_t2 = np.argmax(pred_t2, axis=1)
label_t2 = mat['tst_label_t2']
label_t2 = np.squeeze(label_t2)
'''

oa_t1, kc_t1, cm_t1 = metric(label_t1,pred_t1)
oa_t2, kc_t2, cm_t2 = metric(label_t2,pred_t2)

plt.figure()
plt.subplot('121')
sb.heatmap(cm_t1, cmap='seismic')
plt.title('oa=%.4f, kc=%.4f'%(oa_t1,kc_t1))
plt.subplot('122')
sb.heatmap(cm_t2, cmap='seismic')
plt.title('oa=%.4f, kc=%.4f'%(oa_t2,kc_t2))