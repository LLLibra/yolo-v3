import torch
import torch.nn as nn
import math

## VGG16
## 输入为224*224的图像 VGG的输入224*224的RGB图像，预处理就是每一个像素减去了均值。
# 改进点总结：
# 1.使用了更小的3*3的卷积核代替5*5的卷及和，2个3*3等于5*5,3个3*3等于7*7,还减少了参数增加了网络深度。
# 2.尝试加入1*1的卷积核,在不影响输入维度的情况下,引入非线性变换。（vgg16有一个版本有1*1,其他都为3*3）。
# 3.训练时可以先训练浅层的VGG网络，然后将该参数用于深层的VGGNet,加快收敛。
# 4.VGGNet使用了Multi-Scale的方法做数据增强，将原始图像缩放到不同尺寸S，然后再随机裁切224′224的图片。

##备注：并没有实现1*1的版本 



class VGG_Net(nn.Module):

    def __init__(self,features,num_classes=1000,PATH=None):
        super(VGG_Net,self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes),
        )

        if PATH==None:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x



def make_layers(cfg,batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d,nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'vgg11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def VGG11(**kwargs):
    model = VGG_Net(make_layers(cfg['vgg11'],batch_norm=True),**kwargs)
    return model


def VGG13(**kwargs):
    model = VGG_Net(make_layers(cfg['vgg13'],batch_norm=True),**kwargs)
    return model

def VGG16(**kwargs):
    model = VGG_Net(make_layers(cfg['vgg16'],batch_norm=True),**kwargs)
    return model

def VGG19(**kwargs):
    model = VGG_Net(make_layers(cfg['vgg19'],batch_norm=True),**kwargs)
    return model


if __name__ == '__main__':
    input = torch.rand(1,3,224,224)
    net = VGG16(num_classes=101)
    net_paprms = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("parameters_count:", net_paprms)
    outputs = net.forward(input)
    print(outputs.size())





