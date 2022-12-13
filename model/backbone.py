from pathlib import Path

import torch
from torch import nn

from element import Conv

ROOT = Path(__file__).resolve().parents[1]



class Darknet19(nn.Module):
    def __init__(self, depthwise=False):
        super().__init__()
        self.conv1 = Conv(3, 32, kernel_size=3, padding=1, depthwise=depthwise)
        self.conv2 = Conv(32, 64, kernel_size=3, padding=1, depthwise=depthwise)
        self.conv3 = nn.Sequential(
            Conv(64, 128, kernel_size=3, padding=1, depthwise=depthwise),
            Conv(128, 64, kernel_size=1, depthwise=depthwise),
            Conv(64, 128, kernel_size=3, padding=1, depthwise=depthwise)
        )
        self.conv4 = nn.Sequential(
            Conv(128, 256, kernel_size=3, padding=1, depthwise=depthwise),
            Conv(256, 128, kernel_size=1, depthwise=depthwise),
            Conv(128, 256, kernel_size=3, padding=1, depthwise=depthwise)
        )
        self.conv5 = nn.Sequential(
            Conv(256, 512, kernel_size=3, padding=1, depthwise=depthwise),
            Conv(512, 256, kernel_size=1, depthwise=depthwise),
            Conv(256, 512, kernel_size=3, padding=1, depthwise=depthwise),
            Conv(512, 256, kernel_size=1, depthwise=depthwise),
            Conv(256, 512, kernel_size=3, padding=1, depthwise=depthwise)
        )
        self.conv6 = nn.Sequential(
            Conv(512, 1024, kernel_size=3, padding=1, depthwise=depthwise),
            Conv(1024, 512, kernel_size=1, depthwise=depthwise),
            Conv(512, 1024, kernel_size=3, padding=1, depthwise=depthwise),
            Conv(1024, 512, kernel_size=1, depthwise=depthwise),
            Conv(512, 1024, kernel_size=3, padding=1, depthwise=depthwise),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)


    def forward(self, x):
        out = self.pool(self.conv1(x))
        out = self.pool(self.conv2(out))
        out = self.pool(self.conv3(out))
        C4 = self.conv5(self.pool(self.conv4(out)))
        C5 = self.conv6(self.pool(C4))
        return (C4, C5)


def build_backbone(depthwise=False):
    feat_dims = (512, 1024)
    model = Darknet19(depthwise=depthwise)
    
    if depthwise:
        ckpt = torch.load(ROOT / "weights" / "darknet19_depthwise.pt")
    else:
        ckpt = torch.load(ROOT / "weights" / "darknet19.pt")
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model, feat_dims



if __name__ == "__main__":
    input_size = 416
    device = torch.device('cpu')
    backbone, feat_dims = build_backbone(depthwise=False)

    x = torch.randn(1, 3, input_size, input_size).to(device)
    ftrs = backbone(x)
    for ftr in ftrs:
        print(ftr.shape)