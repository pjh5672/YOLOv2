import cv2
import torch
import numpy as np
from pathlib import Path
from dataloader import Dataset, Augmentation, BaseTransform
from utils import visualize_target, generate_random_color


ROOT = Path(__file__).resolve().parents[0]
MEAN = 0.406, 0.456, 0.485 # BGR
STD = 0.225, 0.224, 0.229 # BGR


def unnormalize(image, mean=MEAN, std=STD):
    image *= std
    image += mean
    image *= 255.
    return image.astype(np.uint8)







if __name__ == "__main__":
    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 416

    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    class_list = train_dataset.class_list
    color_list = generate_random_color(len(class_list))
    train_transformer = Augmentation(size=input_size)
    train_dataset.load_transformer(transformer=train_transformer)

    index = -1
    filename, input_tensor, label, shapes = train_dataset[index]
    image = unnormalize(input_tensor)
    image_with_bbox = visualize_target(image=input_tensor, label=label.numpy(), 
                                       class_list=class_list, color_list=color_list)

    print(filename, input_tensor.shape, label)
    cv2.imwrite(f'./asset/aug_{filename}', image_with_bbox)
