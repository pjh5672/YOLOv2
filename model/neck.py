import torch
from torch import nn


class PassthroughLayer(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride


    def forward(self, ftrs):
        C4, C5 = ftrs
        h, w = C4.shape[-2:]
        h_div, w_div = h // self.stride, w // self.stride
        C4_11 = C4[..., :h_div, :w_div]
        C4_12 = C4[..., :h_div, w_div:]
        C4_21 = C4[..., h_div:, :w_div]
        C4_22 = C4[..., h_div:, w_div:]
        return torch.cat((C5, C4_11, C4_12, C4_21, C4_22), dim=1)


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