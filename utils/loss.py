import sys
from pathlib import Path

import torch
from torch import nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import set_grid



class YoloLoss():
    def __init__(self, grid_size, anchors):
        self.lambda_noobj = 0.5
        self.lambda_coord = 5.0
        self.num_boxes = 5
        self.iou_thres = 0.5
        self.grid_size = grid_size
        self.num_attributes = 1 + 4 + 1 + 1 
        self.obj_loss_func = nn.MSELoss(reduction='none')
        self.box_loss_func = nn.MSELoss(reduction='none')
        self.cls_loss_func = nn.CrossEntropyLoss(reduction='none')
        grid_x, grid_y = set_grid(grid_size=self.grid_size)
        self.grid_x = grid_x.contiguous().view((1, -1, 1))
        self.grid_y = grid_y.contiguous().view((1, -1, 1))
        self.anchors = anchors


    def __call__(self, predictions, labels):
        self.device = predictions.device
        self.bs = predictions.shape[0]
        targets = self.build_batch_target(labels).to(self.device)

        with torch.no_grad():
            iou_pred_with_target = self.calculate_iou(pred_box_cxcywh=predictions[..., 1:5], target_box_cxcywh=targets[..., 1:5])

        pred_obj = predictions[..., 0]
        pred_box_txty = predictions[..., 1:3]
        pred_box_twth = predictions[..., 3:5]
        pred_cls = predictions[..., 5:].permute(0, 3, 1, 2)

        target_obj = (targets[..., 0] == 1).float()
        target_noobj = (targets[..., 6] == 0).float()
        target_box_txty = targets[..., 1:3]
        target_box_twth = targets[..., 3:5]
        target_cls = targets[..., 5].long()

        obj_loss = self.obj_loss_func(pred_obj, iou_pred_with_target) * target_obj
        obj_loss = obj_loss.sum() / self.bs

        noobj_loss = self.obj_loss_func(pred_obj, target_noobj * 0) * target_noobj
        noobj_loss = noobj_loss.sum() / self.bs

        txty_loss = self.box_loss_func(pred_box_txty, target_box_txty).sum(dim=-1) * target_obj
        txty_loss = txty_loss.sum() / self.bs

        twth_loss = self.box_loss_func(pred_box_twth, target_box_twth).sum(dim=-1) * target_obj
        twth_loss = twth_loss.sum() / self.bs

        cls_loss = self.cls_loss_func(pred_cls, target_cls) * target_obj
        cls_loss = cls_loss.sum() / self.bs

        multipart_loss = obj_loss + self.lambda_noobj * noobj_loss + self.lambda_coord * (txty_loss + twth_loss) + cls_loss
        return multipart_loss, obj_loss, noobj_loss, txty_loss, twth_loss, cls_loss


    def calculate_iou_target_with_anchors(self, target_wh, anchor_wh):
        w1, h1 = target_wh
        w2, h2 = anchor_wh.t()
        inter = torch.min(w1, w2) * torch.min(h1, h2)
        union = (w1 * h1) + (w2 * h2) - inter
        return inter/union


    def build_target(self, label):
        target = torch.zeros(size=(self.grid_size, self.grid_size, self.num_boxes, self.num_attributes), dtype=torch.float32)
        
        if -1 in label[:, 0]:
            return target
        else:
            for item in label:
                cls_id = item[0].long()
                grid_i = (item[1] * self.grid_size).long()
                grid_j = (item[2] * self.grid_size).long()
                ious_target_with_anchor = self.calculate_iou_target_with_anchors(target_wh=item[3:5], anchor_wh=self.anchors)
                best_index = ious_target_with_anchor.max(dim=0).indices
                keep = ious_target_with_anchor >= self.iou_thres

                tx = (item[1] * self.grid_size) - grid_i
                ty = (item[2] * self.grid_size) - grid_j
                tw = torch.log(item[3] / self.anchors[best_index, 0])
                th = torch.log(item[4] / self.anchors[best_index, 1])
                
                target[grid_j, grid_i, best_index, 0] = 1.0
                target[grid_j, grid_i, best_index, 1:5] = torch.tensor([tx, ty, tw, th])
                target[grid_j, grid_i, best_index, 5] = cls_id
                target[grid_j, grid_i, keep, 6] = 1.0
            return target

    
    def build_batch_target(self, labels):
        batch_target = torch.stack([self.build_target(label) for label in labels], dim=0)
        return batch_target.view(self.bs, -1,  self.num_boxes, self.num_attributes)


    def calculate_iou(self, pred_box_cxcywh, target_box_cxcywh):
        pred_box_x1y1x2y2 = self.transform_cxcywh_to_x1y1x2y2(pred_box_cxcywh)
        target_box_x1y1x2y2 = self.transform_cxcywh_to_x1y1x2y2(target_box_cxcywh)

        x1 = torch.max(pred_box_x1y1x2y2[..., 0], target_box_x1y1x2y2[..., 0])
        y1 = torch.max(pred_box_x1y1x2y2[..., 1], target_box_x1y1x2y2[..., 1])
        x2 = torch.min(pred_box_x1y1x2y2[..., 2], target_box_x1y1x2y2[..., 2])
        y2 = torch.min(pred_box_x1y1x2y2[..., 3], target_box_x1y1x2y2[..., 3])
        
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        union = abs(pred_box_cxcywh[..., 2] * pred_box_cxcywh[..., 3]) + abs(target_box_cxcywh[..., 2] * target_box_cxcywh[..., 3]) - inter
        inter[inter.gt(0)] = inter[inter.gt(0)] / union[inter.gt(0)]
        return inter


    def transform_cxcywh_to_x1y1x2y2(self, boxes):
        xc = (boxes[..., 0] + self.grid_x.to(self.device)) / self.grid_size
        yc = (boxes[..., 1] + self.grid_y.to(self.device)) / self.grid_size
        w = torch.exp(boxes[..., 2]) * self.anchors[:, 0].to(self.device)
        h = torch.exp(boxes[..., 3]) * self.anchors[:, 1].to(self.device)
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        return torch.stack((x1, y1, x2, y2), dim=-1)



if __name__ == "__main__":
    from torch import optim
    from torch.utils.data import DataLoader
    
    from dataloader import Dataset, BasicTransform
    from model import YoloModel

    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 416
    batch_size = 4
    device = torch.device('cuda')

    transformer = BasicTransform(input_size=input_size)
    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    train_dataset.load_transformer(transformer=transformer)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True)
    anchors = train_dataset.anchors
    num_classes = len(train_dataset.class_list)

    model = YoloModel(input_size=input_size, num_classes=num_classes, anchors=anchors).to(device)
    criterion = YoloLoss(grid_size=model.grid_size, anchors=model.anchors)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    optimizer.zero_grad()

    for epoch in range(50):
        acc_loss = 0.0
        model.train()
        optimizer.zero_grad()

        for index, minibatch in enumerate(train_loader):
            filenames, images, labels, ori_img_sizes = minibatch
            predictions = model(images.to(device))
            loss = criterion(predictions=predictions, labels=labels)
            loss[0].backward()
            optimizer.step()
            optimizer.zero_grad()

            acc_loss += loss[0].item() # obj_loss, noobj_loss, txty_loss, twth_loss, cls_loss
        print(acc_loss / len(train_loader))