# -*- coding:UTF-8 -*-
import torch
import sys
sys.path.append("..")
import torch.nn as nn
from base_Net.VGG_Net import VGG16_SSD
import torch.nn.init as init
from itertools import product as product
from torch.autograd import Function,Variable
from math import sqrt
import torch.nn.functional as F



##参考链接 https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py#L43
##需要用到VGG

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
    best_prior_overlap.squeeze_(0)
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


###将真实结果 转化为预测结果
def encode(match,prior,variance):


    g_cxcy = (match[:,:2]+match[:,2:])/2-prior[:,:2]

    g_cxcy /= (variance[0]*prior[:,2:])


    g_wh = (match[:,2:]-match[:,:2])/prior[:,2:]
    g_wh = torch.log(g_wh)/variance[1]

    return torch.cat([g_cxcy,g_wh],1)





### 求两个矩形的交集
### 返回[A,B]
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
def jaccard(box_a, box_b):
    inter = intersect(box_a,box_b)

    area_a = ((box_a[:,2]-box_a[:,0])*(box_a[:,3]-box_a[:,1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:,2]-box_b[:,0])*(box_b[:,3]-box_b[:,1])).unsqueeze(0).expand_as(inter)

    union = area_a + area_b -inter

    return inter/union


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


###损失函数
class SSD_Loss(nn.Module):

    def __init__(self,num_classes,overlap_thresh,prior_for_matching,
                 bkg_label,neg_mining,neg_pos,neg_overlap,encode_target,
                 use_gpu=True):
        super(SSD_Loss,self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.variance = voc['variance']
        self.negpos_ratio = neg_pos ##正负样本比


    def forward(self,preds,targets):
        ##conf_data (batch_size,num_priors,num_classes)
        ##loc_data (batch_size,num_priors,4)
        ##priors (num_priors,4)
        ##targets (batch_size,num_objs,5)
        loc_data, conf_data , priors = preds

        num = loc_data.size(0)
        priors = priors[:loc_data.size(1),:]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        loc_t = torch.tensor(num,num_priors,4)
        conf_t = torch.LongTensor(num,num_priors)

        for idx in range(num):
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            defaults = priors.data

            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        loc_t = Variable(loc_t,requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        #背景的置信度为0
        pos = conf_t>0
        num_pos = pos.sum(dim=1,keepdim=True)


        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p,loc_t,size_average=False)



        batch_conf = conf_data.view(-1,self.num_classes)
        loss_c = log_sum_exp(batch_conf)-batch_conf.gather(1,conf_t.view(-1,1))


        ##Hard Negative Mining
        loss_c[pos]=0
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

        return  loss_l,loss_c



## 将预测结果转化为真实结果
def decode(loc,priors,variances):

    boxes = torch.cat((priors[:, :2]+loc[:,:2]*variances[0]*priors[:,2:],
                       priors[:, 2:]*torch.exp(loc[:,2:]*variances[1])),1)

    boxes[:, :2] -= boxes[:,2:]/2
    boxes[:, 2:] += boxes[:,:2]/2
    return boxes


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




# 测试时使用
# 1.将网络的预测(预测结果+先验框)将转化为图片上的像素框
# 2.设置置信度阈值筛选
# 3.使用nms进行筛选
class Detect(Function):
    def __init__(self,num_classes,bkg_label,top_k,conf_thresh,nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh

        self.conf_thresh = conf_thresh
        self.variance = voc['variance']

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
    'steps':[8,16,32,64,100,300],
    'min_sizes':[30,60,111,162,216,264],
    'max_sizes':[60,111,162,213,264,315],
    'aspect_ratio':[[2],[2,3],[2,3],[2,3],[2],[2]],
    'variance':[0.1,0.2],
    'clip':True,
    'name':'VOC',
}


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


















































































































