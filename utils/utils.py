# -*- coding:UTF-8 -*-
import torch
import torch.nn as nn
from base_Net.VGG_Net import VGG16_SSD
import torch.nn.init as init
from itertools import product as product
from torch.autograd import Function,Variable
from math import sqrt



### 求两个矩形的交集
### 返回[A,B]
##box_a  [矩形个数,x1y1x2y2]
def intersect(box_a,box_b):
    A = box_a.size(0)
    B = box_b.size(0)

    ##两个右下角
    max_xy = torch.min(box_a[:,2:].unsqueeze(1).expand(A,B,2),
                       box_b[:,2:].unsqueeze(0).expand(A,B,2))

    ##两个左上角
    min_xy = torch.max(box_a[:,:2].unsqueeze(1).expand(A,B,2),
                       box_b[:,:2].unsqueeze(0).expand(A,B,2))

    inter = torch.clamp((max_xy-min_xy),min=0)

    return inter[:,:,0]*inter[:,:,1]



####等同于计算两个框的IoU
##a为真实值 b为预测值
##box_a  [矩形个数,x1y1x2y2]
def jaccard(box_a, box_b):
    inter = intersect(box_a,box_b)

    area_a = ((box_a[:,2]-box_a[:,0])*(box_a[:,3]-box_a[:,1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:,2]-box_b[:,0])*(box_b[:,3]-box_b[:,1])).unsqueeze(0).expand_as(inter)

    union = area_a + area_b -inter

    return inter/union

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


####将x1,y1,x2,y2转换为x,y,w,h
def center_size(boxes):
    return torch.cat((boxes[:,2:]+boxes[:,:2])/2,
                      boxes[:,2:]-boxes[:,:2],1)


###将x,y,w,h 转换为x1,y1,x2,y2
def point_form(boxes):
    return torch.cat((boxes[:,:2]-boxes[:,2:]/2,
                      boxes[:,:2]+boxes[:,2:]/2),1)



###用于非常小的数值求和
def log_sum_exp(x):

    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


###boxes [矩形个数,x1y1x2y2]
def nms(boxes,scores,overlap=0.5,top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = torch.mul(x2-x1,y2-y1)
    v,idx = scores.sort(0)

    idx = idx[-top_k:]

    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()

    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel()>0:
        i = idx[-1]
        keep[count]=i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]

        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        xx1 = torch.clamp(xx1,min=x1[i])
        yy1 = torch.clamp(yy1,min=y1[i])
        xx2 = torch.clamp(xx2,max=x2[i])
        yy2 = torch.clamp(yy2,max=y2[i])

        w.resize_as_(xx2)
        h.resize_as_(yy2)

        inter_area = w*h

        rem_area = torch.index_select(area,0,idx)
        union = (rem_area-inter_area)+area[i]
        IoU = inter_area/union

        idx = idx[IoU.le(overlap)]

    return keep,count