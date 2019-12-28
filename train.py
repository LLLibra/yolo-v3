# -*- coding:UTF-8 -*-
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import argparse

from Dataset.COCODataset import get_COCODataLoader
from Config.train_config import config as train_config
##from Config.ssd_config import voc as config


from utils.yolo_loss import *
from Model.YOLO_V3 import YOLO_V3
from Config.yolo_config import config

###建立记录文件
log_file = open(train_config["log_file_path"],'w')
def print_out(str):
    print(str)
    log_file.write(str)


#创建数据集
data_loader = get_COCODataLoader(batch_size=train_config["train_batch_size"],
                                 num_workers=train_config["num_workers"],
                                 img_size=config['img_size'],
                                 xywh=config["xywh"])

#创建网络
#model = build_SSD('train',size=300,num_classes=train_config["num_classes"])
model = YOLO_V3(config)

#多GPU训练
# if torch.cuda.is_available():
#     model = nn.DataParallel(model,device_ids=train_config["gpus"])
#     model = model.cuda(device = train_config["gpus"][0])


#导入模型
Model_Path = None
if Model_Path!=None:
    model.load_weights(Model_Path)


#创建优化器
optimizer = optim.SGD(model.parameters(), lr=train_config["lr"],weight_decay=train_config["weight_decay"])
lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, train_config["lr_decay_base_num"], last_epoch=-1)



#创建损失函数

# loss = SSD_Loss(num_classes=train_config["num_classes"],
#                 overlap_thresh=0.5,
#                 prior_for_matching=True,
#                 bkg_label=0,
#                 neg_mining=True,
#                 neg_pos_ratio=3,
#                 neg_overlap=0.5,
#                 encode_target=False,
#                 variance=config["variance"],
#                 gpu_id=train_config["gpus"][0],
#                 use_gpu=False)
loss = create_yolov3_loss(config)




model.train()
# epoch_size = len(dataset) // args.batch_size


#开始训练
for e in range(train_config["train_epoch"]):
    for (step,(images,targets)) in enumerate(data_loader):

        # images = images.cuda(train_config["gpus"][0])

        pred = model(images)

        optimizer.zero_grad()

        ###tot_loss = loss(pred,targets) ##SSD损失函数
        tot_loss = tot_yolo_loss(pred,targets,loss) ###yolo v3 loss


        tot_loss.backward()
        optimizer.step()
        print("step:", step)
        if step % train_config["lr_decay_epoch"] == 0 and step > 0:
            print("step:", step)
            lr_decay.step()
            print_out("lr:{:.8f}\n".format(lr_decay.get_lr()[0]))
            # print('one_epoch_time:', float((time.time() - st_time) / lr_decay_epoch * len(train_DataLoader) / 3600),
            #       ' hour')
            # st_time = time.time()
    if e %train_config["save_epoch"]==0 and e>0:
        torch.save(model.state_dict(), train_config["save_path"])

##问题1：dataloader sample
##问题2：box_iou
##问题3：x,y别弄反了


















































