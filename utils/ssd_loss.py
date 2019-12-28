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

from torch.autograd import Function,Variable
from utils import *
import torch.nn.functional as F

## 将预测结果转化为真实结果
def decode(loc,priors,variances):

    boxes = torch.cat((priors[:, :2]+loc[:,:2]*variances[0]*priors[:,2:],
                       priors[:, 2:]*torch.exp(loc[:,2:]*variances[1])),1)

    boxes[:, :2] -= boxes[:,2:]/2
    boxes[:, 2:] += boxes[:,:2]/2
    return boxes

###将真实结果 转化为预测结果
def encode(match,prior,variance):


    g_cxcy = (match[:,:2]+match[:,2:])/2-prior[:,:2]

    g_cxcy /= (variance[0]*prior[:,2:])


    g_wh = (match[:,2:]-match[:,:2])/prior[:,2:]
    g_wh = torch.log(g_wh)/variance[1]

    return torch.cat([g_cxcy,g_wh],1)



##匹配一张图 里面的所有可能预测框和ground truth
def match(threshold,truth,priors,variances,label,loc_t,conf_t,idx):

    ###truth [num_obj,4]
    ###prior [num_priors,4]
    ###variance [num_priors,4]
    ###label [num_obj]
    overlaps = jaccard(truth,point_form(priors))

    #为每个ground truth找到IoU最大的预测框
    best_prior_overlap,best_prior_idx = overlaps.max(1,keepdim=True)
    #为每个预测框找到IoU最大的ground truth
    best_truth_overlap,best_truth_idx = overlaps.max(0,keepdim=True)

    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    ##确保是匹配上的框不会因为IoU值不够而被过滤了
    best_truth_overlap.index_fill_(0,best_prior_idx,2)

    #确定每个预测框对应的哪个ground truth
    #首先优先 每个ground truth和它IoU最大的预测框对应,其次才是每个没匹配上的预测框和它IoU最大的Ground Truth对应
    #但是有个漏洞,当两个ground truth和同一个预测框的IoU最大时  无法处理
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]]=j

    matches = truth[best_truth_idx]    #[num_priors,4]
    conf = label[best_truth_idx] + 1   #[num_priors] 背景为0 所以其他类别+1

    conf[best_truth_overlap<threshold]=0   #小于阈值则为背景
    loc = encode(matches,priors,variances)
    loc_t[idx] = loc
    conf_t[idx] = conf

###损失函数
class SSD_Loss(nn.Module):

    def __init__(self,num_classes,overlap_thresh,prior_for_matching,
                 bkg_label,neg_mining,neg_pos_ratio,neg_overlap,encode_target,variance,
                 gpu_id,use_gpu=True):
        super(SSD_Loss,self).__init__()
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.variance = variance
        self.negpos_ratio = neg_pos_ratio ##正负样本比


    def forward(self,preds,targets):
        ##conf_data (batch_size,num_priors,num_classes)
        ##loc_data (batch_size,num_priors,4)
        ##priors (num_priors,4)
        ##targets (batch_size,num_objs,5)
        ###target应该是以百分比的形式 x1,y1,x2,y2格式
        loc_data, conf_data , priors = preds

        num = loc_data.size(0)
        priors = priors[:loc_data.size(1),:]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        loc_t = torch.Tensor(num,num_priors,4)
        conf_t = torch.LongTensor(num,num_priors)

        for idx in range(num):
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            defaults = priors.data

            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)

        if self.use_gpu:
            loc_t = loc_t.cuda(self.gpu_id)
            conf_t = conf_t.cuda(self.gpu_id)

        loc_t = Variable(loc_t,requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        #背景的置信度为0
        #pos [batch,num_priors]
        pos = conf_t>0
        num_pos = pos.sum(dim=1,keepdim=True)


        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p,loc_t,size_average=False)


        batch_conf = conf_data.view(-1,self.num_classes)
        print("batch_conf:",batch_conf.shape)
        print("conf_t:",conf_t.shape)
        loss_c = log_sum_exp(batch_conf)-batch_conf.gather(1,conf_t.view(-1,1))


        ##Hard Negative Mining
        loss_c[pos.view(-1)]=0
        loss_c = loss_c.view(num,-1)
        _,loss_idx = loss_c.sort(1,descending=True)
        _,idx_rank = loss_idx.sort(1)

        num_pos = pos.long().sum(1,keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos,max=pos.size(1)-1)
        neg = idx_rank<num_neg.expand_as(idx_rank)
        #######


        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p,targets_weighted,size_average=False)

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N

        return  loss_l+loss_c

