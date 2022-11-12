import torch
from torch import nn

from element import Conv, weight_init_kaiming_uniform



class YoloHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.route = Conv(in_channels, 1024, kernel_size=3, padding=1)
        self.detect = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.apply(weight_init_kaiming_uniform)


    def forward(self, x):
        out = self.route(x)
        out = self.detect(out)
        return out



if __name__ == "__main__":
    import torch
    from backbone import build_backbone
    from neck import PassthroughLayer

    input_size = 416
    num_classes = 20
    num_boxes = 5
    num_attributes = (1 + 4 + num_classes)
    device = torch.device('cpu')
    backbone, feat_dims = build_backbone(pretrained=True)
    neck = PassthroughLayer(stride=2)
    head = YoloHead(in_channels=feat_dims[0]*4 + feat_dims[1], out_channels=num_attributes*num_boxes)

    x = torch.randn(1, 3, input_size, input_size).to(device)
    out = backbone(x)
    out = neck(out)
    out = head(out)
    print(out.shape)