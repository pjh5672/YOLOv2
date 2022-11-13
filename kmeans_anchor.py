import os
import pprint
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[0]
SEED = 2023
np.random.seed(SEED)

from dataloader import Dataset
from utils import build_basic_logger, visualize_box_hist



def calcuate_iou(box_wh, clusters):
    ww = np.minimum(clusters[:, 0], box_wh[0])
    hh = np.minimum(clusters[:, 1], box_wh[1])
    
    if np.count_nonzero(ww == 0) > 0 or np.count_nonzero(hh == 0) > 0:
        raise ValueError("Box has no area")

    intersection = ww * hh
    box_area = box_wh[0] * box_wh[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou = intersection / (box_area + cluster_area - intersection)
    return iou


def avg_iou(boxes_wh, clusters):
    return np.mean([np.max(calcuate_iou(boxes_wh[i], clusters)) for i in range(boxes_wh.shape[0])])


def collect_all_boxes_wh(dataset, apply_image=False):
    boxes_wh = []
    for i in tqdm(range(len(dataset)), desc=f"Collecting boxes for all instances...", ncols=115, leave=False):
        _, image, label = dataset.get_GT_item(i)
        if -1 not in label[:, 0]:
            if apply_image:
                img_h, img_w = image.shape[:2]
                box_w = (label[:, 3] * img_w).astype(int)
                box_h = (label[:, 4] * img_h).astype(int)
                boxes_wh.append(np.stack((box_w, box_h), axis=1))
            else:
                boxes_wh.append(label[:, -2:])
    return np.concatenate(boxes_wh, axis=0)


def kmeans_iou(boxes_wh, n_cluster):
    rows = boxes_wh.shape[0]
    distances = np.empty((rows, n_cluster))
    last_clusters = np.zeros((rows,))
    clusters = boxes_wh[np.random.choice(rows, n_cluster, replace=False)]

    while True:
        for i in range(rows):
            distances[i] = 1 - calcuate_iou(boxes_wh[i], clusters)
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break
        
        for j in range(n_cluster):
            clusters[j] = np.median(boxes_wh[nearest_clusters == j], axis=0)
        last_clusters = nearest_clusters

    boxes_info = np.concatenate((boxes_wh, last_clusters[:, np.newaxis]), axis=1)
    return clusters, boxes_info


def parse_args(make_dirs=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Name to log training")
    parser.add_argument("--apply_img", type=str, nargs='?', const=True ,help="Clustering box dimensions with original image size")
    parser.add_argument("--data", type=str, default="toy.yaml", help="Path to data.yaml")
    parser.add_argument("--n_cluster", type=int, default=5, help="Number of clusters")
    args = parser.parse_args()
    args.data = ROOT / "data" / args.data
    args.exp_path = ROOT / 'experiment' / args.exp_name

    if make_dirs:
        os.makedirs(args.exp_path, exist_ok=True)
    return args


def main():
    args = parse_args(make_dirs=True)
    logger = build_basic_logger(args.exp_path / 'anchor.log', set_level=1)
    logger.info(f"[Arguments]\n{pprint.pformat(vars(args))}\n")

    dataset = Dataset(yaml_path=args.data, phase='train')
    boxes_wh = collect_all_boxes_wh(dataset=dataset, apply_image=args.apply_img)
    clusters, boxes_info = kmeans_iou(boxes_wh=boxes_wh, n_cluster=5)
    ratios = [round(item, 2) for item in (clusters[:, 0] / clusters[:, 1])]

    logger.info(f"Avg IOU: {avg_iou(boxes_wh, clusters) * 100:.2f}%")
    logger.info(f"Boxes:\n {clusters}")
    logger.info(f"Ratios: {sorted(ratios)}")

    data_df, box_hist_fig = visualize_box_hist(boxes=boxes_info)
    data_df.to_csv(str(args.exp_path / 'box_info.csv'))
    box_hist_fig.savefig(str(args.exp_path / 'box_hist.jpg'))


if __name__ == "__main__":
    main()
    


    