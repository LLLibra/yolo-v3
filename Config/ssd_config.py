# -*- coding:UTF-8 -*-
from utils.ssd_loss import *
from Model.SSD import  build_SSD
extras = {
    '300': [[1024,256,512],[512,128,256],[256,128,256],[256,128,256]],
    '512': [],
}

mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # 最终特征图中每个点有多少个box
    '512': [],
}

##SSD 300 config
voc = {
    'num_classes': 21,
    'feature_maps':[38,19,10,5,3,1],
    'min_dim':300,
    'img_size':300,
    'xywh':False,
    'steps':[8,16,32,64,100,300],
    'min_sizes':[30,60,111,162,216,264],
    'max_sizes':[60,111,162,213,264,315],
    'aspect_ratio':[[2],[2,3],[2,3],[2,3],[2],[2]],
    'variance':[0.1,0.2],
    'clip':True,
    'name':'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'img_size':300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}