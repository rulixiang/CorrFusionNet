import numpy as np


class config():
    def __init__(self):
        arr = [[128,   0,   0],[128, 128,   0],[  0,   0, 128],[  0, 128,   0],
               [  0, 192,   0],[128, 128, 128],[192, 128,   0],[ 64,   0, 128],
               [128,   0, 128],[192,   0, 128],[128,  64,   0],[192, 128, 128],
               [ 64,  64,   0],[  0,  64, 128],]
        self.colormap = np.array(arr,dtype=np.float32) / np.max(arr)
        self.class_name = ['0-undefined',
            '1-administration', '2-commercial', '3-water', '4-farmland',
            '5-greenspace', '6-transportation', '7-industrial',
            '8-residential-1', '9-residential-2', '10-residential-3',
            '11-road', '12-parking', '13-bareland', '14-playground'
        ]
        self.shape = [190, 237]
        self.dir_t1 = 'D:/lib/wuhan/img_process/2014/sure/'
        self.dir_t2 = 'D:/lib/wuhan/img_process/2016/sure/'
        self.step = 1
