import cv2
from pathlib import Path
from dataloader import Dataset, Augmentation, BaseTransform

ROOT = Path(__file__).resolve().parents[0]

if __name__ == "__main__":
    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 416

    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    train_transformer = Augmentation(size=input_size)
    train_dataset.load_transformer(transformer=train_transformer)

    for index, minibatch in enumerate(train_dataset):
        filename, input_tensor, label, shapes = train_dataset[index]
        print(input_tensor.shape)

    # image = cv2.imread
    print(len(train_dataset))