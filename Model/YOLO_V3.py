# -*- coding:UTF-8 -*-
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from base_Net.DarkNet import darknet53
import numpy as np
import math

##https://github.com/BobLiu20/YOLOv3_PyTorch/blob/master/nets/model_main.py

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou



class YOLO_V3(nn.Module):
    def __init__(self,config):
        super(YOLO_V3,self).__init__()

        self.backbone = darknet53()

        #DarkNet最后四层的输出通道数
        self.backbone_C_out = [128,256,512,1024]


        final_out0 = len(config['yolo']['anchors'][0])*(5+config['yolo']['classes'])
        self.embedding0 = self.make_embedding(self.backbone_C_out[-2:],self.backbone_C_out[-1],final_out0)



        final_out1 = len(config['yolo']['anchors'][1])*(5+config['yolo']['classes'])
        self.embedding0_upsamle = nn.Sequential(
            self.block(512,256,1),
            nn.Upsample(scale_factor=2,mode='nearest'),
        )
        self.embedding1 = self.make_embedding(self.backbone_C_out[-3:-1], self.backbone_C_out[-2]+256, final_out1)


        final_out2 = len(config['yolo']['anchors'][2])*(5+config['yolo']['classes'])
        self.embedding1_upsamle = nn.Sequential(
            self.block(256,128,1),
            nn.Upsample(scale_factor=2,mode='nearest'),
        )
        self.embedding2 = self.make_embedding(self.backbone_C_out[-4:-2], self.backbone_C_out[-3]+128, final_out2)


    def block(self,inchannel,outchannel,kernel_size):

        pad = (kernel_size-1)//2
        return nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=kernel_size,padding=pad,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(0.1),
        )

    def make_embedding(self,C_list,in_channel,final_out):
        return nn.Sequential(
            self.block(in_channel,C_list[0], 1),
            self.block(C_list[0], C_list[1], 3),
            self.block(C_list[1], C_list[0], 1),
            self.block(C_list[0], C_list[1], 3),
            self.block(C_list[1], C_list[0], 1),
            self.block(C_list[0], C_list[1], 3),
            nn.Conv2d(C_list[1],final_out,kernel_size=1,stride=1,padding=0,bias=True)
        )

    def forward(self,x):
        def _branch(_embedding,_in):
            for i,e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in,out_branch

        x2,x1,x0 = self.backbone(x)

        out0,out0_brach = _branch(self.embedding0,x0)

        x1_in = self.embedding0_upsamle(out0_brach)

        x1_in = torch.cat([x1_in,x1],1)
        out1,out1_branch = _branch(self.embedding1,x1_in)

        x2_in = self.embedding1_upsamle(out1_branch)
        x2_in = torch.cat([x2_in,x2],1)
        out2,out2_branch = _branch(self.embedding2,x2_in)
        return  out0,out1,out2



if __name__ == "__main__":
    m = YOLO_V3()
    x = torch.randn(1, 3, 416, 416)
    y0, y1, y2 = m(x)
    print(y0.size())
    print(y1.size())
    print(y2.size())

##(80+5)*3
# torch.Size([1, 255, 13, 13])
# torch.Size([1, 255, 26, 26])
# torch.Size([1, 255, 52, 52])



