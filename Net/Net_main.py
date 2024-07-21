import torch
import torch.nn as nn
import torch.nn.functional as F

from work_4.Net.ResNet import resnet34
from work_4.Net.Swin import Swintransformer
from work_4.Net.DCB import DCB
from work_4.Net.MLWD import MLWD
from work_4.Net.MSA import MSA




# 边缘增强的宽解码器网络
class EEWDNet(nn.Module):
    def __init__(self, args):
        super(EEWDNet, self).__init__()

        self.resnet = resnet34()
        self.swin = Swintransformer(224)

        if args.train_mode:
            self.resnet.load_state_dict(torch.load('/data2021/tb/AllWork/work_3/Model/resnet34.pth'), strict=False)
            self.swin.load_state_dict(torch.load('/data2021/tb/AllWork/work_4/Model/swin224.pth')['model'],strict=False)
        
        self.conv_s1 = nn.Sequential(
                    nn.Conv2d(128, 64, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
        self.conv_s2 = nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
        self.conv_s3 = nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
        
        self.sup_res = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.sup_swin = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

        self.cat_conv1 = nn.Sequential(
                    nn.Conv2d(1024, 512, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                )
        
        self.msa1 = MSA(512)
        self.sup1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.mlwd1 = MLWD(256)

        self.msa2 = MSA(256)
        self.sup2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.dcb1 = DCB(128)
        self.mlwd2 = MLWD(128)

        self.msa3 = nn.Sequential(
                    nn.Conv2d(128, 64, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
        self.sup3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.dcb2 = DCB(64)
        self.mlwd3 = MLWD(64)

        self.sup4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        y = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)

        s1, s2, s3, s4 = self.swin(y)
        rx = self.resnet(x)

        o1 = self.sup_res(rx[3])
        o2 = self.sup_swin(s4)

        s1 = F.interpolate(self.conv_s1(s1), size=(256, 256), mode='bilinear', align_corners=True)
        s2 = F.interpolate(self.conv_s2(s2), size=(128, 128), mode='bilinear', align_corners=True)
        s3 = F.interpolate(self.conv_s3(s3), size=( 64,  64), mode='bilinear', align_corners=True)
        s4 = F.interpolate(s4, size=(32, 32), mode='bilinear', align_corners=True)

        d = self.cat_conv1(torch.cat((rx[3], s4), dim=1))
        d = d + rx[3] + s4

        d = self.msa1(d)
        o3 = self.sup1(d)
        d = F.interpolate(d, size=s3.size()[2:], mode='bilinear', align_corners=True)
        d = self.mlwd1(rx[2], s3, d)

        d = self.msa2(d)
        o4 = self.sup2(d)
        rx[1] = self.dcb1(rx[1])
        d = F.interpolate(d, size=s2.size()[2:], mode='bilinear', align_corners=True)
        d = self.mlwd2(rx[1], s2, d)

        d = self.msa3(d)
        o5 = self.sup3(d)
        rx[0] = self.dcb2(rx[0])
        d = F.interpolate(d, size=s1.size()[2:], mode='bilinear', align_corners=True)
        d = self.mlwd3(rx[0], s1, d)
        
        out = self.sup4(d)
        return out, o1, o2, o3, o4, o5


