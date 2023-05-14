# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 16:05  2023-05-14
from block import *
from collections import OrderedDict
from thop import profile

t0 = [40, 1, 2, 8, 2, "GELU"]
t1 = [64, 1, 2, 8, 2, "GELU"]
t2 = [96, 1, 2, 8, 2, "GELU"]
s = [128, 1, 2, 13, 2, "ReLU"]
m = [144, 3, 4, 18, 3, "ReLU"]
l = [192, 3, 4, 18, 3, "ReLU"]


class FasterNet(nn.Module):
    def __init__(self, parameter_list, class_number):
        super().__init__()
        if parameter_list[-1] == "GELU":
            Acti = nn.GELU
        elif parameter_list[-1] == "ReLU":
            Acti = nn.ReLU
        else:
            Acti = nn.ReLU
            
        channelList = [3]
        channelList += [2 ** i * parameter_list[0] for i in range(4)]
        blockList = parameter_list[1:-1]
        
        self.Embedding1 = Merging(channelList[0], channelList[1], ksize=4, stride=4)
        self.Stage1 = BasicStage(channelList[1], blockList[0], n_div=4,
                                 drop_path=0.5, Acti=Acti)
        
        self.Embedding2 = Merging(channelList[1], channelList[2], ksize=2, stride=2)
        self.Stage2 = BasicStage(channelList[2], blockList[1], n_div=4,
                                 drop_path=0.5, Acti=Acti)
        
        self.Embedding3 = Merging(channelList[2], channelList[3], ksize=2, stride=2)
        self.Stage3 = BasicStage(channelList[3], blockList[2], n_div=4,
                                 drop_path=0.5, Acti=Acti)
        
        self.Embedding4 = Merging(channelList[3], channelList[4], ksize=2, stride=2)
        self.Stage4 = BasicStage(channelList[4], blockList[3], n_div=4,
                                 drop_path=0.5, Acti=Acti)
        
        self.classifier = nn.Sequential(OrderedDict([
            ('global_average_pooling', nn.AdaptiveAvgPool2d(1)),
            ('conv', nn.Conv2d(channelList[4], 1280, kernel_size=1, bias=False)),
            ('act', Acti()),
            ('flat', nn.Flatten()),
            ('fc', nn.Linear(1280, class_number, bias=True))
        ]))
    
    def forward(self, x):
        x1 = self.Embedding1(x)
        x1 = self.Stage1(x1)
        
        x2 = self.Embedding2(x1)
        x2 = self.Stage2(x2)
        
        x3 = self.Embedding3(x2)
        x3 = self.Stage3(x3)
        
        x4 = self.Embedding4(x3)
        x4 = self.Stage4(x4)
        
        out = self.classifier(x4)
        return out


if __name__ == "__main__":
    net = FasterNet(l, 1000)
    a = torch.randn((1, 3, 224, 224))
    b = net(a)
    print(b.shape)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(net, inputs=(input,))
    print('FLOPs = %.2f G ' % ((flops / 1000 ** 3)))
    print('Params = %.2f M' % (params / 1000 ** 2))
