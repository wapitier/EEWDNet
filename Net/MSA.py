import torch
import torch.nn as nn
import torch.nn.functional as F


# 深浅层特征注意力计算
class Attention(nn.Module):
    def __init__(self, in_channel, num_heads=8):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = in_channel // num_heads
        self.scale = head_dim ** -0.5

        self.lnx = nn.LayerNorm(in_channel)
        self.lnz = nn.LayerNorm(in_channel)

        self.qv = nn.Linear(in_channel, in_channel*2)
        self.k = nn.Linear(in_channel, in_channel)

        self.proj = nn.Linear(in_channel, in_channel)

    def forward(self, x, z):
        batch_size, channel, height, width = x.size()

        z = F.interpolate(z, size=(height, width), mode='bilinear')   

        sc = x
        x = x.view(batch_size, channel, -1).permute(0, 2, 1)
        x = self.lnx(x)

        z = z.view(batch_size, channel, -1).permute(0, 2, 1)
        z = self.lnz(z)
        
        B, N, C = x.shape

        x_qv = self.qv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q, x_v = x_qv[0], x_qv[1]

        z_k = self.k(z).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        z_k = z_k[0]
   
        attn = (x_q @ z_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ x_v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
  
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, channel, height, width)
  
        return x + sc


# 多尺度注意力模块
class MSA(nn.Module):
    def __init__(self, in_channel):
        super(MSA, self).__init__()

        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(inplace=True)
                )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(inplace=True)
                )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(inplace=True)
                )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(inplace=True)
                )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.att1 = Attention(in_channel)
        self.att2 = Attention(in_channel)
        self.att3 = Attention(in_channel)

        self.cat_conv = nn.Sequential(
                    nn.Conv2d(in_channel * 2, in_channel // 2, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(in_channel // 2),
                    nn.ReLU(inplace=True)
                )
        

    def forward(self, x): 
        
        x1 = self.conv1(x)
        x1 = self.pool1(x1)

        x2 = self.conv1(x1)
        x2 = self.pool1(x2)

        x3 = self.conv1(x2)
        x3 = self.pool1(x3)

        x4 = self.conv1(x3)
        x4 = self.pool1(x4)

        x3 = self.att1(x3, x4)
        x2 = self.att2(x2, x3)
        x1 = self.att3(x1, x2)

        x1 = F.interpolate(x1, size=x.size()[2:], mode='bilinear')
        x = self.cat_conv(torch.cat((x, x1), dim=1))

        return x
