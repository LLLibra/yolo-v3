import torch
import torch.nn as nn
import math

## VGG16
## 输入为224*224的图像
改进点总结：
1.使用了更小的3*3的卷积核代替5*5的卷及和，2个3*3等于5*5,3个3*3等于7*7,还减少了参数增加了网络深度。
2.尝试加入1*1的卷积核，在不影响数
class VGG_Net(nn.Module):

    def __init__(self,PATH=None):



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









