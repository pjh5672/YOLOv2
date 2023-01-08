from pathlib import Path

import gdown
import torch
from torch import nn

from element import Conv

ROOT = Path(__file__).resolve().parents[1]


model_urls = {
    "darknet19-448": "https://drive.google.com/file/d/1PvEYIms_iU8VSvtmQ9D_MHhJuULA45l4/view?usp=share_link"
}


class DarkNet19(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(3, 32, kernel_size=3, padding=1, act="leaky_relu")
        self.conv2 = Conv(32, 64, kernel_size=3, padding=1, act="leaky_relu")
        self.conv3 = nn.Sequential(
            Conv(64, 128, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(128, 64, kernel_size=1, act="leaky_relu"),
            Conv(64, 128, kernel_size=3, padding=1, act="leaky_relu")
        )
        self.conv4 = nn.Sequential(
            Conv(128, 256, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(256, 128, kernel_size=1, act="leaky_relu"),
            Conv(128, 256, kernel_size=3, padding=1, act="leaky_relu")
        )
        self.conv5 = nn.Sequential(
            Conv(256, 512, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(512, 256, kernel_size=1, act="leaky_relu"),
            Conv(256, 512, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(512, 256, kernel_size=1, act="leaky_relu"),
            Conv(256, 512, kernel_size=3, padding=1, act="leaky_relu")
        )
        self.conv6 = nn.Sequential(
            Conv(512, 1024, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(1024, 512, kernel_size=1, act="leaky_relu"),
            Conv(512, 1024, kernel_size=3, padding=1, act="leaky_relu"),
            Conv(1024, 512, kernel_size=1, act="leaky_relu"),
            Conv(512, 1024, kernel_size=3, padding=1, act="leaky_relu"),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)


    def forward(self, x):
        out = self.pool(self.conv1(x))
        out = self.pool(self.conv2(out))
        out = self.pool(self.conv3(out))
        C4 = self.conv5(self.pool(self.conv4(out)))
        C5 = self.conv6(self.pool(C4))
        return (C4, C5)


def build_backbone():
    feat_dims = (512, 1024)
    model = DarkNet19()
    download_path = ROOT / "weights" / "darknet19.pt"
    if not download_path.is_file():
        gdown.download(model_urls["darknet19-448"], str(download_path), quiet=False, fuzzy=True)
    ckpt = torch.load(download_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model, feat_dims



if __name__ == "__main__":
    input_size = 416
    device = torch.device('cpu')
    backbone, feat_dims = build_backbone()

    x = torch.randn(1, 3, input_size, input_size).to(device)
    ftrs = backbone(x)
    for ftr in ftrs:
        print(ftr.shape)