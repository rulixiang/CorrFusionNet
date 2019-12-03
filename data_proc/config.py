import numpy as np


class config():
    def __init__(self):
        arr = np.array([[0,0,205],[65,105,225],[135,206,235],[0,139,69],[0,216,0],[238,154,73],[163,124,2],[255,38,38],[205,38,38],[139,26,26],[255,231,186],[48,48,48],[179,151,143],[186,85,211]],dtype=np.float32)/255  
        self.colormap = np.array(arr,dtype=np.float32) / np.max(arr)
        self.class_name = ['0-undefined',
            '1-administration', '2-commercial', '3-water', '4-farmland',
            '5-greenspace', '6-transportation', '7-industrial',
            '8-residential-1', '9-residential-2', '10-residential-3',
            '11-road', '12-parking', '13-bareland', '14-playground'
        ]
        self.shape = [190, 237]
        self.dir_t1 = './2014/sure/'
        self.dir_t2 = './2016/sure/'
        self.step = 1
