import torch
from torch import nn

from element import Conv



class YoloHead(nn.Module):
    def __init__(self, in_channels, out_channels, depthwise=False):
        super().__init__()
        self.conv = Conv(in_channels, 1024, kernel_size=3, padding=1, depthwise=depthwise)
        self.detect = nn.Conv2d(1024, out_channels, kernel_size=1)


    def forward(self, x):
        out = self.conv(x)
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
    backbone, feat_dims = build_backbone(depthwise=False)
    neck = PassthroughLayer(in_channels=feat_dims, depthwise=False)
    head = YoloHead(in_channels=neck.feat_dims[0]*4+neck.feat_dims[1], out_channels=num_attributes * num_boxes, depthwise=False)

    x = torch.randn(1, 3, input_size, input_size).to(device)
    out = backbone(x)
    out = neck(out)
    out = head(out)
    print(out.shape)