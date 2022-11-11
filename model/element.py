from torch import nn


def weight_init_kaiming_uniform(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1.0)
        module.bias.data.fill_(0.0)


class Conv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride=1, padding=0, dilation=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)
