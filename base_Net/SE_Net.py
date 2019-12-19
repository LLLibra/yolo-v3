# -*- coding:UTF-8 -*-
import torch
from torch import nn
import math

### 是一个网络子结构 用于插入网络

### SE Net 用于ResNet  对残差进行处理
###

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) ##自适应池化 输入NCHW 指定输出特征图大小
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        ##全局平均池化，batch和channel和原来一样保持不变
        y = self.avg_pool(x).view(b, c)

        ##全连接层+池化
        y = self.fc(y).view(b, c, 1, 1)

        ##和原特征图相乘
        return x * y.expand_as(x)  ## y原来是NC y.expand_as(x)即将NC复制扩展为NCHW




if __name__ == '__main__':
    net = SELayer(channel=101)
    net_paprms = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("parameters_count:",net_paprms)
    input = torch.randn([1,64,224,224])
    out = net(input)
    print(out.shape)














































