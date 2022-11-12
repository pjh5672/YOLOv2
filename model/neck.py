import torch
from torch import nn


class PassthroughLayer(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride


    def forward(self, ftrs):
        c4, c5 = ftrs
        h, w = c4.shape[-2:]
        h_div, w_div = h // self.stride, w // self.stride
        c4_11 = c4[..., :h_div, :w_div]
        c4_12 = c4[..., :h_div, w_div:]
        c4_21 = c4[..., h_div:, :w_div]
        c4_22 = c4[..., h_div:, w_div:]
        return torch.cat((c5, c4_11, c4_12, c4_21, c4_22), dim=1)



if __name__ == "__main__":
    from backbone import build_backbone

    input_size = 416
    device = torch.device('cpu')
    backbone, feat_dims = build_backbone(pretrained=True)
    neck = PassthroughLayer(stride=2)

    x = torch.randn(1, 3, input_size, input_size).to(device)
    ftrs = backbone(x)
    out = neck(ftrs)
    print(out.shape)