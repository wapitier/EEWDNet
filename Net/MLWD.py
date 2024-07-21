import torch
import torch.nn as nn
import torch.nn.functional as F


# 宽卷积模块
class WideConv(nn.Module):
    def __init__(self, in_channel):
        super(WideConv, self).__init__()

        self.branch0 = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1),
                )
        self.branch1 = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(inplace=True)
                )
        self.branch2 = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=7, stride=1, padding=3),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(inplace=True)
                )
        self.branch3 = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, dilation=3, padding=3),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(inplace=True)
                )
        

    def forward(self, x):
        x1 = self.branch0(x)
        x2 = self.branch1(x) 
        x3 = self.branch2(x) 
        x4 = self.branch3(x) 

        return x1 + x2 + x3 + x4




# 多级宽解码器模块
class MLWD(nn.Module):
    def __init__(self, in_channel):
        super(MLWD, self).__init__()

        self.cat_conv = nn.Sequential(
                    nn.Conv2d(in_channel * 3, in_channel, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(inplace=True)
                )
        
        self.wdc1 = WideConv(in_channel)
        self.wdc2 = WideConv(in_channel)

        self.conv_mid = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(inplace=True)
                )

        self.conv_end = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(inplace=True)
                )


    def forward(self, r, s, d):
        B, C, H, W = r.size()
        
        d = F.interpolate(d, size=(H, W), mode='bilinear')

        x = torch.cat((r, s, d), dim=1)

        x = self.cat_conv(x)

        x = x + self.wdc1(x)
        
        x = self.conv_mid(x)

        x = x + self.wdc2(x)

        x = x + self.conv_end(x)

        return x
