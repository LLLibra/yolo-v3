# -*- coding:UTF-8 -*-
import torch
from torch import nn
import math


###第一代： 使用了 1*1 3*3 5*5 增加了模型宽度  后又在3*3 和5*5的卷积前增加1*1的卷积减少参数
###第二代： 加入了BN层 用两个3*3 代替了5*5
###第三代： 使用1*7和7*1代替的7*7 1*3和3*1代替了3*3
###第四代： 加入了short——cut也就是残差的方法 用于提升速度
###
###该模型为第三代 输入为224*224

###pre->A*3->B->C*4->D->E*2->fc


###A结构
###1*1
###1*1+5*5
###1*1+3*3+3*3
###pool+1*1
###四个concat

def base_conv(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs),
        nn.BatchNorm2d(out_channels, eps=0.001),
    )


class InceptionA(nn.Module):
    def __init__(self,input_size,pool_feature):
        super(InceptionA,self).__init__()
        self.branch1 = base_conv(in_channels=input_size,out_channels=64,kernel_size=1)

        self.branch2 = nn.Sequential(
            base_conv(in_channels=input_size,out_channels=48,kernel_size=1),
            base_conv(in_channels=48, out_channels=64, kernel_size=5,padding=2),
        )

        self.branch3 = nn.Sequential(
            base_conv(in_channels=input_size, out_channels=64, kernel_size=1),
            base_conv(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            base_conv(in_channels=96, out_channels=96, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=input_size, out_channels=pool_feature, kernel_size=1)
        )

    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x4 = self.branch4(x)

        outputs = [x1,x2,x3,x4]

        return torch.cat(outputs,1)



###
###3*3
###1*1+3*3+3*3
###pool
class InceptionB(nn.Module):
    def __init__(self,input_size):
        super(InceptionB,self).__init__()
        self.branch1 = base_conv(in_channels=input_size,out_channels=384,kernel_size=3,stride=2)
        self.branch2 = nn.Sequential(
            base_conv(in_channels=input_size,out_channels=64,kernel_size = 1),
            base_conv(in_channels=64,out_channels=96,kernel_size = 3,padding = 1),
            base_conv(in_channels=96, out_channels=96, kernel_size=3,stride=2),
        )
        self.branch3 = nn.AvgPool2d(kernel_size=3,stride=2)


    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        outputs = [x1,x2,x3]

        return torch.cat(outputs,1)





###
###1*1
###1*1+1*7+7*1
###1*1+7*1+1*7+7*1+1*7
###pool+1*1
###
class InceptionC(nn.Module):
    def __init__(self,input_size,C_channels):
        super(InceptionC,self).__init__()
        self.branch1 = base_conv(in_channels=input_size,out_channels=192,kernel_size=1)

        self.branch2 = nn.Sequential(
            base_conv(in_channels=input_size, out_channels=C_channels, kernel_size=1),
            base_conv(in_channels=C_channels, out_channels=C_channels, kernel_size=(1,7), padding=(0,3)),
            base_conv(in_channels=C_channels, out_channels=192, kernel_size=(7,1), padding=(3,0))
        )

        self.branch3 = nn.Sequential(
            base_conv(in_channels=input_size, out_channels=C_channels, kernel_size=1),
            base_conv(in_channels=C_channels, out_channels=C_channels, kernel_size=(7, 1), padding=(3, 0)),
            base_conv(in_channels=C_channels, out_channels=C_channels, kernel_size=(1, 7), padding=(0, 3)),
            base_conv(in_channels=C_channels, out_channels=C_channels, kernel_size=(7, 1), padding=(3, 0)),
            base_conv(in_channels=C_channels, out_channels=192, kernel_size=(1, 7), padding=(0, 3)),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            base_conv(in_channels=input_size, out_channels=192, kernel_size=1),
        )




    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        outputs = [x1,x2,x3,x4]

        return torch.cat(outputs,1)



###1*1+3*3
###1*1+1*7+7*1+3*3
###pool
class InceptionD(nn.Module):
    def __init__(self,input_size):
        super(InceptionD, self).__init__()
        self.branch1 = nn.Sequential(
            base_conv(in_channels=input_size, out_channels=192, kernel_size=1),
            base_conv(in_channels=192, out_channels=320, kernel_size=3,stride=2),
        )

        self.branch2 = nn.Sequential(
            base_conv(in_channels=input_size, out_channels=192, kernel_size=1),
            base_conv(in_channels=192, out_channels=192, kernel_size=(1,7),padding=(0,3)),
            base_conv(in_channels=192, out_channels=192, kernel_size=(7,1), padding=(3,0)),
            base_conv(in_channels=192, out_channels=192, kernel_size=3, stride=2),
        )

        self.branch3 = nn.AvgPool2d(kernel_size=3, stride=2)



    def forward(self,x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        outputs = [x1,x2,x3]

        return torch.cat(outputs,1)



###
###比较复杂
###
class InceptionE(nn.Module):
    def __init__(self,input_size):
        super(InceptionE, self).__init__()
        self.branch_1 = base_conv(in_channels=input_size, out_channels=320, kernel_size=1)

        self.branch_2 = base_conv(in_channels=input_size, out_channels=384, kernel_size=1)
        self.branch_2_1 = base_conv(in_channels=384, out_channels=384, kernel_size=(1,3),padding=(0,1))
        self.branch_2_2 = base_conv(in_channels=384, out_channels=384, kernel_size=(3,1),padding=(1,0))


        self.branch_3 = nn.Sequential(
            base_conv(in_channels=input_size, out_channels=448, kernel_size=1),
            base_conv(in_channels=448, out_channels=384, kernel_size=3,padding=1),
        )
        self.branch_3_1 = base_conv(in_channels=384, out_channels=384, kernel_size=(1,3),padding=(0,1))
        self.branch_3_2 = base_conv(in_channels=384, out_channels=384, kernel_size=(3,1),padding=(1,0))

        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            base_conv(in_channels=input_size, out_channels=192, kernel_size=1)
        )

    def forward(self,x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x2_1 = self.branch_2_1(x2)
        x2_2 = self.branch_2_2(x2)
        x2 = torch.cat([x2_1,x2_2],1)
        x3 = self.branch_3(x)
        x3_1 = self.branch_3_1(x3)
        x3_2 = self.branch_3_2(x3)
        x3 = torch.cat([x3_1,x3_2],1)

        x4 = self.branch_4(x)
        outputs = [x1,x2,x3,x4]

        return torch.cat(outputs,1)



###
###准备利用中间层的特征
###pool+1*1+5*5+fc
###https://zhuanlan.zhihu.com/p/30172532 有时会用到这个
def InceptionAux(input_size,num_classes):
    fc = nn.Linear(768,num_classes)
    fc.stddev = 0.001
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=5, stride=1),
        base_conv(in_channels=input_size,out_channels=128,kernel_size=1),
        base_conv(in_channels=128, out_channels=768, kernel_size=5),
        fc
    )


class Inception_v3(nn.Module):
    def __init__(self,class_num=1000):
        super(Inception_v3, self).__init__()
        self.pre = nn.Sequential(
            base_conv(in_channels=3,out_channels=32,kernel_size=3,stride=2),
            base_conv(in_channels=32, out_channels=32, kernel_size=3),
            base_conv(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            base_conv(in_channels=64, out_channels=80, kernel_size=1),
            base_conv(in_channels=80, out_channels=192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.body = nn.Sequential(
            InceptionA(192, pool_feature=32),
            InceptionA(256, pool_feature=64),
            InceptionA(288, pool_feature=64),
            InceptionB(288),
            InceptionC(768, C_channels=128),
            InceptionC(768, C_channels=160),
            InceptionC(768, C_channels=160),
            InceptionC(768, C_channels=192),
            InceptionD(768),
            InceptionE(1280),
            InceptionE(2048)
        )
        self.drop = nn.Dropout(0.5)
        self.classifier = nn.Linear(2048, class_num)
        self._initialize_weights()

    def forward(self,x):
        x = self.pre(x)
        x = self.body(x)
        print(x.shape)
        x = nn.AvgPool2d(kernel_size=8)(x)  # kernel_size为7是因为经过多次下采样之后feature map的大小为7*7，即224->112->56->28->14->7
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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


if __name__ == "__main__":
    net = Inception_v3(class_num=101)
    net_paprms = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("parameters_count:",net_paprms)
    input = torch.randn([1,3,224,224])
    out = net(input)
    print(out.shape)























































































































