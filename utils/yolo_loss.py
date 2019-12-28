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
import math

from utils import *
from utils import bbox_iou

def create_yolov3_loss(config):
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"])))
    return yolo_losses


def tot_yolo_loss(outputs,labels,yolo_losses):
    losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
    losses = []
    for _ in range(len(losses_name)):
        losses.append([])
    for i in range(3):
        _loss_item = yolo_losses[i](outputs[i], labels)
        for j, l in enumerate(_loss_item):
            losses[j].append(l)
    losses = [sum(l) for l in losses]
    loss = losses[0]
    return loss


class YOLOLoss(nn.Module):
    def __init__(self,anchors,num_classes,img_size,use_cuda=False):
        super(YOLOLoss,self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5+num_classes
        self.img_size = img_size

        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.use_cuda = use_cuda

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()


    def forward(self,input,target=None):
        ###target应该是以百分比的形式 xywh格式
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)


        stride_h = self.img_size[1]/in_h
        stride_w = self.img_size[0]/in_w

        scaled_anchors = [(a_w/stride_w,a_h/stride_h) for a_w,a_h in self.anchors]

        pred = input.view(bs,self.num_anchors,self.bbox_attrs,in_h,in_w).permute(0,1,3,4,2).contiguous()

        x = torch.sigmoid(pred[...,0])
        y = torch.sigmoid(pred[...,1])
        w = pred[...,2]
        h = pred[...,3]
        conf = torch.sigmoid(pred[...,4])
        pred_cls = torch.sigmoid(pred[...,5:])


        if target is not None:
            mask,noobj_mask,tx,ty,tw,th,tconf,tcls = self.get_target(target,scaled_anchors,in_w,in_h,self.ignore_threshold)

            if self.use_cuda:
                mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
                tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
                tconf, tcls = tconf.cuda(), tcls.cuda()

            loss_x = self.bce_loss(x * mask, tx * mask)
            loss_y = self.bce_loss(y * mask, ty * mask)
            loss_w = self.mse_loss(w * mask, tw * mask)
            loss_h = self.mse_loss(h * mask, th * mask)
            loss_conf = self.bce_loss(conf * mask, mask) + \
                        0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)

            loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])


            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                   loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                   loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), \
                   loss_h.item(), loss_conf.item(), loss_cls.item()


        else:
            ##将网络输出值 decode 为真实值
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
                bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
                bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)


            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)


            pred_boxes = FloatTensor(pred[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h


            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data



    ## encode 将COCO中的数据(真实值)修改为可以求损失的形式(网络输出)
    def get_target(self,target,anchors,in_w,in_h,ignore_threshold):
        ###target应该是以百分比的形式 xywh格式
        bs = len(target)

        mask = torch.zeros(bs,self.num_anchors,in_h,in_w,requires_grad=False)
        noobj_mask = torch.ones(bs,self.num_anchors,in_h,in_w,requires_grad=False)
        tx = torch.zeros(bs,self.num_anchors,in_h,in_w,requires_grad=False)
        ty = torch.zeros(bs,self.num_anchors,in_h,in_w,requires_grad=False)
        tw = torch.zeros(bs,self.num_anchors,in_h,in_w,requires_grad=False)
        th = torch.zeros(bs,self.num_anchors,in_h,in_w,requires_grad=False)

        tconf = torch.zeros(bs,self.num_anchors,in_h,in_w,requires_grad=False)
        tcls = torch.zeros(bs,self.num_anchors,in_h,in_w,self.num_classes,requires_grad=False)
        for b in range(bs):
            target_ = target[b]
            for t in range(target_.shape[0]):
                if target_[t].sum() ==0 :
                    continue
                gx = target_[t,0]*in_w
                gy = target_[t,1]*in_h
                gw = target_[t,2]*in_w
                gh = target_[t,3]*in_h


                gi = int(gx)
                gj = int(gy)

                gt_box = torch.FloatTensor(np.array([0,0,gw,gh])).unsqueeze(0)

                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors,2)),np.array(anchors)),1))

                anch_ious = bbox_iou(gt_box,anchor_shapes)##gt_box 为xywh


                noobj_mask[b,anch_ious>ignore_threshold,gj,gi] = 0

                best_n = np.argmax(anch_ious)   ###每个点有最好的一个anchor

                mask[b, best_n, gj, gi] = 1

                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj


                tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)

                tconf[b, best_n, gj, gi] = 1

                tcls[b, best_n, gj, gi, int(target_[t, 4])] = 1

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls