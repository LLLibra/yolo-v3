# -*- coding:UTF-8 -*-
import torch
from torch import nn
import math

##输入为224*224


##ResNet18 和 ResNet34使用的
##3*3的卷积
class ResidualBlock(nn.Module):
    expansion = 1 ##输入和输出的通道差多少倍
    def __init__(self, inchannel, outchannel, stride, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),  # 要采样的话在这里改变stride
            nn.BatchNorm2d(outchannel),  # 批处理正则化
            nn.ReLU(inplace=True),  # 激活
            nn.Conv2d(outchannel, outchannel*self.expansion, kernel_size=3, stride=1, padding=1, bias=False),  # 采样之后注意保持feature map的大小不变
            nn.BatchNorm2d(outchannel*self.expansion),
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)  # 计算残差
        out += residual
        return nn.ReLU(inplace=True)(out)  # 注意激活


##ResNet50 ResNet101 和 ResNet152 用的
##1*1->3*3->1*1
class ResidualBlock_v2(nn.Module):
    expansion = 4
    def __init__(self, inchannel, outchannel, stride, shortcut=None):
        super(ResidualBlock_v2, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),  # 要采样的话在这里改变stride
            nn.BatchNorm2d(outchannel),  # 批处理正则化
            nn.ReLU(inplace=True),  # 激活
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),  # 采样之后注意保持feature map的大小不变
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel*self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(outchannel*self.expansion),
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)  # 计算残差
        out += residual
        return nn.ReLU(inplace=True)(out)  # 注意激活

# ResNet类
class ResNet(nn.Module):
    def __init__(self,cfg,base_block,class_num=1000):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2, padding=1),
        )
        self.input_size = 64
        self.base_block = base_block
        self.body = self.make_layers(cfg)  # 具有重复模块的部分
        self.classifier = nn.Linear(self.base_block.expansion*512, class_num)  # 末尾的部分
        self._initialize_weights()

    def make_layers(self, blocklist):  # 注意传入列表而不是解列表
        self.layers = []
        for index, blocknum in enumerate(blocklist):
            shortcut = nn.Sequential(
                nn.Conv2d(self.input_size,self.base_block.expansion*64*2**index,kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(self.base_block.expansion*64*2**index)
            )  # 使得输入输出通道数调整为一致
            self.layers.append(self.base_block(inchannel=self.input_size , outchannel=64*2**index,stride=2, shortcut=shortcut))  # 每次变化通道数时进行下采样
            self.input_size = self.base_block.expansion*64*2**index
            for i in range(1, blocknum): #除了第一层不需要short_cut 从开始,其他的层都从开始
                print(index,"-",i,":",self.input_size)
                self.layers.append(self.base_block(inchannel= self.input_size, outchannel=64*2**index, stride=1))
                self.input_size = self.base_block.expansion*64*2**index
        return nn.Sequential(*self.layers)


    def _initialize_weights(self):
        for m in self.modules():
            # print(type(m))
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def forward(self, x):
        print(x.shape)
        x = self.pre(x)
        print(x.shape)
        x = self.body(x)
        print(x.shape)
        x = nn.AvgPool2d(7)(x)  # kernel_size为7是因为经过多次下采样之后feature map的大小为7*7，即224->112->56->28->14->7
        x = x.view(x.size(0), -1)
        #return input
        x = self.classifier(x)
        return x



def ResNet18(class_num):
    return  ResNet(cfg=[2,2,2,2],base_block=ResidualBlock,class_num=class_num)

def ResNet34(class_num):
    return ResNet(cfg=[3,4,6,3],base_block=ResidualBlock,class_num=class_num)

def ResNet50(class_num):
    return ResNet(cfg=[3,4,6,3],base_block=ResidualBlock_v2,class_num=class_num)

def ResNet101(class_num):
    return ResNet(cfg=[3,4,23,3],base_block=ResidualBlock_v2,class_num=class_num)

def ResNet152(class_num):
    return ResNet(cfg=[3,8,36,30],base_block=ResidualBlock_v2,class_num=class_num)



if __name__ == "__main__":
    net = ResNet101(class_num=101)
    net_paprms = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("parameters_count:",net_paprms)
    input = torch.randn([1,3,224,224])
    out = net(input)
    print(out.shape)





