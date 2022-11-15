import torch
from torch import nn

from element import Conv, weight_init_kaiming_uniform



class PassthroughLayer(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
        self.conv1 = Conv(512, 64, kernel_size=1)
        self.conv2 = nn.Sequential(
            Conv(1024, 1024, kernel_size=3, padding=1),
            Conv(1024, 1024, kernel_size=3, padding=1),
        )

    def forward(self, ftrs):
        C4, C5 = ftrs
        C4 = self.conv1(C4)
        bs, c, h, w = C4.shape
        h_div, w_div = h // self.stride, w // self.stride

        C4 = C4.view(bs, c, h_div, self.stride, w_div, self.stride).transpose(3, 4).contiguous()
        C4 = C4.view(bs, c, h_div * w_div, self.stride * self.stride).transpose(2, 3).contiguous()
        C4 = C4.view(bs, c, self.stride * self.stride, h_div, w_div).transpose(1, 2).contiguous()
        C4 = C4.view(bs, -1, h_div, w_div)
        C5 = self.conv2(C5)
        return torch.cat((C4, C5), dim=1)


if __name__ == "__main__":
    from backbone import build_backbone

    input_size = 416
    device = torch.device('cpu')
    backbone, feat_dims = build_backbone(arch_name='darknet19', pretrained=True)
    neck = PassthroughLayer(stride=2)

    x = torch.randn(1, 3, input_size, input_size).to(device)
    ftrs = backbone(x)
    out = neck(ftrs)
    print(out.shape)