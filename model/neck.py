import torch
from torch import nn

from element import Conv, weight_init_kaiming_uniform


class PassthroughLayer(nn.Module):
    def __init__(self, in_channels, stride=2):
        super().__init__()
        self.stride = stride
        P4_dims = 512 * 4
        P5_dims = 1024
        self.ftr_dims = P4_dims + P5_dims
        self.conv1 = Conv(in_channels[0]*4, P4_dims, 1)
        self.conv2 = Conv(in_channels[1], P5_dims, 1)
        self.apply(weight_init_kaiming_uniform)


    def forward(self, ftrs):
        C4, C5 = ftrs
        h, w = C4.shape[-2:]
        h_div, w_div = h // self.stride, w // self.stride
        C4_11 = C4[..., :h_div, :w_div]
        C4_12 = C4[..., :h_div, w_div:]
        C4_21 = C4[..., h_div:, :w_div]
        C4_22 = C4[..., h_div:, w_div:]
        P4 = self.conv1(torch.cat((C4_11, C4_12, C4_21, C4_22), dim=1))
        P5 = self.conv2(C5)
        return torch.cat((P4, P5), dim=1)



if __name__ == "__main__":
    from backbone import build_backbone

    input_size = 416
    device = torch.device('cpu')
    backbone, feat_dims = build_backbone(arch_name='darknet19', pretrained=True)
    neck = PassthroughLayer(in_channels=feat_dims, stride=2)

    x = torch.randn(1, 3, input_size, input_size).to(device)
    ftrs = backbone(x)
    out = neck(ftrs)
    print(out.shape)