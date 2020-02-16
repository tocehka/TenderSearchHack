import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BaseConv2d, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block(x)


class Inception(nn.Module):

    def __init__(self):
        super(Inception, self).__init__()
        self.branch1x1 = BaseConv2d(128, 32, kernel_size=1, padding=0)
        self.branch1x1_2 = BaseConv2d(128, 32, kernel_size=1, padding=0)
        self.branch3x3_reduce = BaseConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3 = BaseConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_reduce_2 = BaseConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3_2 = BaseConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_3 = BaseConv2d(32, 32, kernel_size=3, padding=1)
  
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
    
        branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch1x1_2 = self.branch1x1_2(branch1x1_pool)
        
        branch3x3_reduce = self.branch3x3_reduce(x)
        branch3x3 = self.branch3x3(branch3x3_reduce)
        
        branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
        branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
        branch3x3_3 = self.branch3x3_3(branch3x3_2)
        
        outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
        return torch.cat(outputs, 1) + x
    

class SELayer(nn.Module):
    def __init__(self, inplanes, squeeze_ratio=8, activation=nn.PReLU, size=None):
        super(SELayer, self).__init__()
        if size is not None:
            self.global_avgpool = nn.AvgPool2d(size)
        else:
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, int(inplanes / squeeze_ratio), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(int(inplanes / squeeze_ratio), inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out
    
    
class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.
    
    
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, outp_size=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.inv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU(),

            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 3, stride, 1,
                      groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU(),

            nn.Conv2d(in_channels * expand_ratio, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            #SELayer(out_channels, 8, nn.PReLU, outp_size)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.inv_block(x)

        return self.inv_block(x)