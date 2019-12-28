import torch
import torch.nn as nn
import math

##https://github.com/BobLiu20/YOLOv3_PyTorch/blob/master/nets/backbone/darknet.py

class BasicBlock(nn.Module):
    def __init__(self,inchannel,cfg):
        super(BasicBlock,self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(inchannel, cfg[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(cfg[0]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cfg[1]),
            nn.LeakyReLU(0.1),
        )

    def forward(self,x):
        residual = x
        out =  self.body(x)
        out += x

        return out

class Darknet(nn.Module):
    def __init__(self,cfg):
        super(Darknet,self).__init__()
        self.input = 32
        self.pre = nn.Sequential(
            nn.Conv2d(3, self.input, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.input),
            nn.LeakyReLU(0.1),
        )


        self.layer1 = self.make_layers([32,64],cfg[0])
        self.layer2 = self.make_layers([64,128],cfg[1])
        self.layer3 = self.make_layers([128,256],cfg[2])
        self.layer4 = self.make_layers([256,512],cfg[3])
        self.layer5 = self.make_layers([512,1024],cfg[4])

        self._initialize_weights()


    def make_layers(self,cfg,epoch):
        layers=[]
        self.layer_first = nn.Sequential(
            nn.Conv2d(self.input,cfg[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg[1]),
            nn.LeakyReLU(0.1),
        )
        layers.append(self.layer_first)
        self.input= cfg[1]
        for i in range(epoch):
            layers.append(BasicBlock(self.input,cfg))

        return nn.Sequential(*layers)




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
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        return x3,x4,x5


def darknet21(**kwargs):
    return Darknet([1,1,2,2,1])


def darknet53(**kwargs):
    return Darknet([1,2,8,8,4])




if __name__ == '__main__':
    Net = darknet53()
    input = torch.rand(10,3,416,416)
    out1,out2,out3 = Net(input)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)

# torch.Size([10, 256, 52, 52])
# torch.Size([10, 512, 26, 26])
# torch.Size([10, 1024, 13, 13])




