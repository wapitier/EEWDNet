import torch
import torch.nn as nn
import torch.nn.functional as F


# 中心差分卷积 Center Difference Conv
class CDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=True):
        super(CDC, self).__init__()

        padding = dilation if padding is None else padding
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weights_c = self.weight.sum(dim=[2, 3], keepdim=True)
        yc = F.conv2d(x, weights_c, stride=self.stride, padding=0, groups=self.groups)
        y = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return y - yc
    

# 环形差分卷积 Angular Difference Conv
class ADC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=True):
        super(ADC, self).__init__()

        padding = dilation if padding is None else padding
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        shape = self.weight.shape
        weights = self.weight.view(shape[0], shape[1], -1)
        weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape) # clock-wise
        y = F.conv2d(x, weights_conv, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return y


# 径向差分卷积 Radial Difference Conv
class RDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=True):
        super(RDC, self).__init__()

        padding = 2 * dilation if padding is None else padding
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        shape = self.weight.shape
        if self.weight.is_cuda:
            buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
        else:
            buffer = torch.zeros(shape[0], shape[1], 5 * 5)
        weights = self.weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
        buffer[:, :, 12] = 0
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        y = F.conv2d(x, buffer, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return y


# 差分卷积模块
class DCB(nn.Module):
    def __init__(self, in_channel):
        super(DCB, self).__init__()

        self.CDC = CDC(in_channel, in_channel)
        self.ADC = ADC(in_channel, in_channel)
        self.RDC = RDC(in_channel, in_channel)
        self.TCV = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)

        self.cat_conv = nn.Sequential(
                    nn.Conv2d(4 * in_channel, in_channel, kernel_size=3, stride=1, dilation=1, padding=1),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(inplace=True)
                )

        

    def forward(self, x): 
        
        x1 = self.CDC(x)
        x2 = self.ADC(x)
        x3 = self.RDC(x)
        x4 = self.TCV(x)

        out = x + self.cat_conv(torch.cat((x1, x2, x3, x4), dim=1))

        return out
