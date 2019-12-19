# -*- coding:UTF-8 -*-
import torch
from torch import nn
import math

##输入是224*224
# 1、减轻了vanishing-gradient（梯度消失）
# 2、加强了feature的传递
# 3、更有效地利用了feature
# 4、一定程度上较少了参数数量

class DenseLayer(nn.Module):
    def __init__(self,input_size,growth_rate,bn_size):
        super(DenseLayer,self).__init__()
        self.basic = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_size,out_channels=bn_size*growth_rate,kernel_size=1,stride=1),
            nn.BatchNorm2d(bn_size*growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size*growth_rate,out_channels=growth_rate,kernel_size=3,stride=1,padding=1)
        )
        self.drop = nn.Dropout(0.5)

    def forward(self,x):
        new_features = self.basic(x)
        new_features = self.drop(new_features)

        return torch.cat([x,new_features],1)


##growth_rate 是每小层一层增加多少通道
class DenseNet(nn.Module):
    def __init__(self,cfg,input_size,growth_rate=32,bn_size=4,num_classes=1000):
        super(DenseNet,self).__init__()
        self.input_size = input_size
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=self.input_size,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(self.input_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.body = self.make_layers(cfg,bn_size,growth_rate)
        self.classifier=nn.Linear(self.input_size,num_classes)
        self._initialize_weights()


    def make_layers(self,cfg,bn_size,growth_rate):
        layers = []
        for index, num_layers in enumerate(cfg):
            layers.append(self.DenseBlock(num_layers, self.input_size, bn_size, growth_rate))
            self.input_size += num_layers*growth_rate
            if index !=len(cfg)-1:
                layers.append(self.Transition(input_size=self.input_size,out_size=self.input_size//2))
                self.input_size = self.input_size//2
        layers.append(nn.BatchNorm2d(self.input_size))
        return nn.Sequential(*layers)



    def Transition(self, input_size, out_size):
        return nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_size, out_size, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )


    def DenseBlock(self, num_layers, input_size, bn_size, growth_rate):
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(input_size+i*growth_rate, growth_rate, bn_size))
        return nn.Sequential(*layers)


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

    def forward(self,x):
        x = self.pre(x)
        x = self.body(x)
        x = nn.AvgPool2d(7)(x)  # kernel_size为7是因为经过多次下采样之后feature map的大小为7*7，即224->112->56->28->14->7
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def DenseNet121(**kwargs):
    return DenseNet(cfg=[6,12,24,16],input_size=64,growth_rate=32,**kwargs)


def DenseNet169(**kwargs):
    return DenseNet(cfg=[6,12,32,32],input_size=64,growth_rate=32,**kwargs)


def DenseNet201(**kwargs):
    return DenseNet(cfg=[6,12,48,32],input_size=64,growth_rate=32,**kwargs)



def DenseNet161(**kwargs):
    return DenseNet(cfg=[6,12,36,24],input_size=96,growth_rate=48,**kwargs)


if __name__ == "__main__":
    net = DenseNet161(num_classes=101)
    net_paprms = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("parameters_count:",net_paprms)
    input = torch.randn([1,3,224,224])
    out = net(input)
    print(out.shape)








