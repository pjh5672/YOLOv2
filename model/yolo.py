import sys
from pathlib import Path

import gdown
import torch
from torch import nn

from backbone import build_backbone
from neck import PassthroughLayer
from head import YoloHead

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import set_grid


model_urls = {
    "yolov2": "https://drive.google.com/file/d/16q3Hkhz8S8_Sn3IUju9U5z9y2hb-3UKP/view?usp=share_link",
}


class YoloModel(nn.Module):
    def __init__(self, input_size, num_classes, anchors, pretrained=False):
        super().__init__()
        self.stride = 32
        self.num_boxes = 5
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_attributes = 1 + 4 + num_classes
        self.backbone, feat_dims = build_backbone()
        self.neck = PassthroughLayer(in_channels=feat_dims)
        self.head = YoloHead(in_channels=self.neck.feat_dims[0]*4+self.neck.feat_dims[1], 
                             out_channels=self.num_attributes*self.num_boxes)
        self.anchors = torch.tensor(anchors)
        self.set_grid_xy(input_size=input_size)

        if pretrained:
            download_path = ROOT / "weights" / "yolov2.pt"
            if not download_path.is_file():
                gdown.download(model_urls[f"yolov2"], str(download_path), quiet=False, fuzzy=True)
            ckpt = torch.load(download_path, map_location="cpu")
            self.load_state_dict(ckpt["model_state"], strict=False)


    def forward(self, x):
        self.device = x.device
        bs = x.shape[0]

        out = self.backbone(x)
        out = self.neck(out)
        out = self.head(out)
        out = out.permute(0, 2, 3, 1).flatten(1, 2).view((bs, -1, self.num_boxes, self.num_attributes))

        pred_obj = torch.sigmoid(out[..., [0]])
        pred_box_txty = torch.sigmoid(out[..., 1:3])
        pred_box_twth = out[..., 3:5]
        pred_cls = out[..., 5:]

        if self.training:
            return torch.cat((pred_obj, pred_box_txty, pred_box_twth, pred_cls), dim=-1)
        else:
            pred_box = self.transform_pred_box(torch.cat((pred_box_txty, pred_box_twth), dim=-1))
            pred_score = pred_obj * torch.softmax(pred_cls, dim=-1)
            pred_score, pred_label = pred_score.max(dim=-1)
            pred_out = torch.cat((pred_score.unsqueeze(-1), pred_box, pred_label.unsqueeze(-1)), dim=-1)
            return pred_out.flatten(1, 2)


    def transform_pred_box(self, pred_box):
        xc = (pred_box[..., 0] + self.grid_x.to(self.device)) / self.grid_size
        yc = (pred_box[..., 1] + self.grid_y.to(self.device)) / self.grid_size
        w = torch.exp(pred_box[..., 2]) * self.anchors[:, 0].to(self.device)
        h = torch.exp(pred_box[..., 3]) * self.anchors[:, 1].to(self.device)
        return torch.stack((xc, yc, w, h), dim=-1)


    def set_grid_xy(self, input_size):
        self.grid_size = input_size // self.stride
        grid_x, grid_y = set_grid(grid_size=self.grid_size)
        self.grid_x = grid_x.contiguous().view((1, -1, 1))
        self.grid_y = grid_y.contiguous().view((1, -1, 1))



if __name__ == "__main__":
    input_size = 416
    num_classes = 1
    inp = torch.randn(2, 3, input_size, input_size)
    device = torch.device("cpu")

    anchors = [[0.47070834, 0.7668643 ],
               [0.6636637,  0.274     ],
               [0.875,      0.61066663],
               [0.8605263,  0.8736842 ],
               [0.283375,   0.5775    ]]

    model = YoloModel(input_size=input_size, num_classes=num_classes, anchors=anchors).to(device)
    model.train()
    out = model(inp.to(device))
    print(model.grid_size)
    print(out.shape)

    model.eval()
    out = model(inp.to(device))
    print(out.shape)

    model.train()
    inp = torch.randn(2, 3, 608, 608)
    model.set_grid_xy(input_size=608)
    out = model(inp.to(device))
    print(model.grid_size)
    print(out.shape)

    model.eval()
    out = model(inp.to(device))
    print(out.shape)