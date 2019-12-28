# -*- coding:UTF-8 -*-
import torch
import sys
import torch.nn as nn
from base_Net.VGG_Net import VGG16_SSD
import torch.nn.init as init
from itertools import product as product
from torch.autograd import Function,Variable
from math import sqrt
import torch.nn.functional as F


from Config.ssd_config import *
from utils.utils import nms
from utils.ssd_loss import decode


##参考链接 https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py#L43
##需要用到VGG


# 测试时使用
# 1.将网络的预测(预测结果+先验框)将转化为图片上的像素框
# 2.设置置信度阈值筛选
# 3.使用nms进行筛选
class Detect(Function):
    def __init__(self,num_classes,bkg_label,top_k,conf_thresh,nms_thresh,variance):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh

        self.conf_thresh = conf_thresh
        self.variance = variance #voc['variance']

    def forward(self, loc_data, conf_data, prior_data):
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        output = torch.zeros(num,self.num_classes,self.top_k,5)
        conf_pred = conf_data.view(num,num_priors,self.num_classes).transpose(2,1)

        for i in range(num):
            decode_boxes = decode(loc_data[i],prior_data,self.variance)
            conf_scores = conf_pred[i].clone()

            for cl in range(1,self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0 :
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decode_boxes)
                boxes = decode_boxes[l_mask].view(-1,4)

                ids, count = nms(boxes,scores,self.nms_thresh,self.top_k)

                output[i,cl,:count] = torch.cat((scores[ids[:count]].unsqueeze(1),boxes[ids[:count]]),1)
        flt = output.contiguous.view(num,-1,5)
        _,idx = flt[:,:,0].sort(1,descending=True)
        _,rank = idx.sort(1)

        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)




##SSD 的PriorBox
class PriorBox(object):
    def __init__(self,cfg):
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratio']) ##长方形框的个数的一半(因为长宽可以换)

        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']

        self.steps = cfg['steps']
        self.aspect_ratio = cfg['aspect_ratio']
        self.clip = cfg['clip']
        self.version = cfg['name']

    def forward(self):
        mean = []
        for k,f in enumerate(self.feature_maps):
            for i,j in product(range(f), repeat=2):
                f_k = self.image_size/self.steps[k]

                ##先验框的中心
                cx = (j+0.5)/f_k
                cy = (i+0.5)/f_k

                ##两个正方形
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx,cy,s_k,s_k]

                s_k_prime = sqrt(s_k*(self.max_sizes[k]/self.image_size))
                mean += [cx,cy,s_k_prime,s_k_prime]

                for ar in self.aspect_ratio[k]:
                    mean += [cx,cy,s_k*sqrt(ar),s_k/sqrt(ar)]
                    mean += [cx,cy,s_k/sqrt(ar),s_k*sqrt(ar)]

        output = torch.tensor(mean).view(-1,4)
        if self.clip:
            output.clamp_(max=1,min=0)
        return output


##放置于VGG conv4之后
class L2Norm(nn.Module):
    def __init__(self,in_channel,scale):
        super(L2Norm,self).__init__()
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.tensor([in_channel]))
        init.constant_(self.weight,self.gamma)

    def forward(self,x):
        norm = x.pow(2).sum(dim=1,keepdim=True).sqrt()+self.eps
        x = torch.div(x,norm)
        result = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)*x

        return result


class SSD(nn.Module):
    def __init__(self,phase,base,extra,head,num_classes):
        super(SSD,self).__init__()
        self.phase = phase
        self.num_classes = num_classes


        self.priobox = PriorBox(voc)
        self.priors = self.priobox.forward()

        self.vgg =  nn.Sequential(*base)
        self.L2Norm = L2Norm(512.0,20)
        self.extras = nn.Sequential(*extra)

        self.loc = nn.Sequential(*head[0])
        self.conf = nn.Sequential(*head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=1)
            self.detect = Detect(num_classes)


    def forward(self,x):

        loc = [] #loc结果
        conf = [] #conf 结果
        sources =[] ##要放入head（loc conf）处理的数据


        for k in range(23):
           x = self.vgg[k](x)

        x1 = self.L2Norm(x)
        sources.append(x1)

        for k in range(23,len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        for k,v in enumerate(self.extras):
            x = nn.ReLU(inplace=True)(v(x))
            if k%2 == 1:
                sources.append(x)

        for (x,l,c) in zip(sources,self.loc,self.conf):
            loc.append(l(x).permute(0,2,3,1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())


        loc = torch.cat([o.view(o.size(0),-1) for o in loc],1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)


        if self.phase=="train":
            output = (
                loc.view(loc.size(0),-1,4),
                conf.view(conf.size(0),-1,self.num_classes),
                self.priors,
            )
        else:
            output = self.detect(
                loc.view(loc.size(0),-1,4),
                self.softmax(conf.view(conf.size(0),-1,self.num_classes)),
                self.priors.type(type(x.data)),
            )


        return output


###VGG16 之后的延伸
def add_extras(cfg,in_channel):
    layers = []
    for k,c_num in enumerate(cfg):
        c1, c2, c3 = c_num
        if k<2:
            layers.append(nn.Conv2d(c1, c2, kernel_size=1, stride=1))
            layers.append(nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1))
        else:
            layers.append(nn.Conv2d(c1, c2, kernel_size=1, stride=1))
            layers.append(nn.Conv2d(c2, c3, kernel_size=3, stride=1))
    return layers


###用于最终的分类和定位
def add_head(vgg,extra_net,cfg,num_classes):
    loc_layers = []
    conf_layers = []

    ##为vgg conv4的输出准备的
    loc_layers += [nn.Conv2d(vgg[21].out_channels,cfg[0]*4,kernel_size=3,padding=1)]
    conf_layers += [nn.Conv2d(vgg[21].out_channels,cfg[0]*num_classes,kernel_size=3,padding=1)]

    ##为vgg pool5的输出准备的
    loc_layers += [nn.Conv2d(vgg[-2].out_channels,cfg[1]*4,kernel_size=3,padding=1)]
    conf_layers += [nn.Conv2d(vgg[-2].out_channels,cfg[1]*num_classes,kernel_size=3,padding=1)]

    ##为extra_net的各个阶段的输出
    for k,v in enumerate(extra_net[1::2],2):
        loc_layers += [nn.Conv2d(v.out_channels,cfg[k]*4,kernel_size=3,padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]*num_classes, kernel_size=3, padding=1)]

    return (loc_layers,conf_layers)


##目前只支持SSD 300
def build_SSD(phase,size=300,num_classes=21):
    base_vgg = VGG16_SSD()
    extra_net = add_extras(extras[str(size)],1024)               #VGG之后延伸的网络
    head_net = add_head(base_vgg,extra_net,mbox[str(size)],num_classes)

    return SSD(phase,base_vgg,extra_net,head_net,num_classes)

if __name__ == '__main__':
    model = build_SSD('train', size=300, num_classes=101)
    inputs = torch.rand(30,3,300,300)
    pred = model(inputs)
    for i in pred:
        print(i.shape)

    # (30, 8732, 4)
    # (30, 8732, 101)
    # (8732, 4)


















































































































