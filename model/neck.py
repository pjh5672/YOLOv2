import torch
from torch import nn

from element import Conv



class PassthroughLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stride = 2
        self.feat_dims = (64, 1024)
        self.conv1 = Conv(in_channels[0], self.feat_dims[0], kernel_size=1, act="leaky_relu")
        self.conv2 = nn.Sequential(
            Conv(in_channels[1], 1024, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(1024, self.feat_dims[1], kernel_size=3, padding=1, act="leaky_relu"),
        )
        

    def forward(self, ftrs):
        C4, C5 = ftrs
        P4 = self.conv1(C4)
        P5 = self.conv2(C5)
        bs, c, h, w = P4.shape
        h_div, w_div = h // self.stride, w // self.stride
        P4 = P4.view(bs, c, h_div, self.stride, w_div, self.stride).transpose(3, 4).contiguous()
        P4 = P4.view(bs, c, h_div * w_div, self.stride * self.stride).transpose(2, 3).contiguous()
        P4 = P4.view(bs, c, self.stride * self.stride, h_div, w_div).transpose(1, 2).contiguous()
        P4 = P4.view(bs, -1, h_div, w_div)
        return torch.cat((P4, P5), dim=1)



if __name__ == "__main__":
    from backbone import build_backbone

    input_size = 416
    device = torch.device('cpu')
    backbone, feat_dims = build_backbone()
    neck = PassthroughLayer(in_channels=feat_dims)

    x = torch.randn(1, 3, input_size, input_size).to(device)
    ftrs = backbone(x)
    out = neck(ftrs)
    print(out.shape)