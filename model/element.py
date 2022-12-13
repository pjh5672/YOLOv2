import torch
from torch import nn
import torch.nn.functional as F



class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))



class Conv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride=1, padding=0, dilation=1, act=True, depthwise=False):
        super().__init__()
        if depthwise:
            self.conv = nn.Sequential(
                ### Depth-wise ###
                nn.Conv2d(c1, c1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                Mish() if act else nn.Identity(),
                ### Point-wise ###
                nn.Conv2d(c1, c2, kernel_size=1, stride=stride, padding=0, dilation=dilation, groups=1, bias=False),
                nn.BatchNorm2d(c2),
                Mish() if act else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                ### General Conv ###
                nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=False),
                nn.BatchNorm2d(c2),
                Mish() if act else nn.Identity(),
            )

    def forward(self, x):
        return self.conv(x)