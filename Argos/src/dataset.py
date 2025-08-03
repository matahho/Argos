import json
import logging
import os
import torch
from PIL import Image
from torch.utils.data import Subset, random_split, DataLoader
from torchvision.transforms import Compose, ToTensor
from src.settings import DATASET_SIZE, TRAIN_PERCENTAGE, TEST_PERCENTAGE


def collate_fn(batch):
    # batch is a list of (image_tensor, target_dict)
    images, targets = zip(*batch)
    return list(images), list(targets)


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partitioned MTSD data dynamically based on client ID."""
    full_dataset = MTSDDataset(root_dir="./data")


    max_dataset_size = DATASET_SIZE
    if max_dataset_size > len(full_dataset):
        logging.warning(f"max_dataset_size ({max_dataset_size}) exceeds dataset size ({len(full_dataset)})")
        max_dataset_size = len(full_dataset)

    full_dataset = Subset(full_dataset, list(range(max_dataset_size)))


    total_size = len(full_dataset)
    partition_size = total_size // num_partitions
    remainder = total_size % num_partitions

    lengths = [partition_size] * num_partitions
    for i in range(remainder):
        lengths[i] += 1

    subsets = random_split(full_dataset, lengths, generator=torch.Generator().manual_seed(42))

    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(f"Invalid partition_id {partition_id}; must be in range [0, {num_partitions - 1}]")

    partition_dataset = subsets[partition_id]

    assert TRAIN_PERCENTAGE + TEST_PERCENTAGE <= 1.0 , "Test and Train percentage sum must be less than 1.0."

    train_size = int(TRAIN_PERCENTAGE * len(partition_dataset))
    val_size = len(partition_dataset) - train_size

    train_subset, val_subset = random_split(
        partition_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def extract_label_mapping(classes_file):
    label_map = {}
    try:
        with open(classes_file, 'r') as f:
            json_data = json.load(f)
            for class_name, details in json_data.items():
                if "classIndex" in details:
                    label_map[details["classIndex"]] = class_name
    except FileNotFoundError:
        print(f"Error: Classes file not found at {classes_file}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {classes_file}. Check file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return label_map




class MTSDDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir,
            images_dir="images",
            annotations_dir="txts (YOLO)",
            transform=Compose([ToTensor()])
    ):
        self.root_directory = root_dir
        self.transform = transform
        self.images_names = sorted(
            os.listdir(
                os.path.join(self.root_directory, images_dir)
            )
        )
        self.full_images_directory = os.path.join(self.root_directory, images_dir)
        self.full_annotations_directory = os.path.join(self.root_directory, annotations_dir)

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        file_name = self.images_names[idx]
        img_path = os.path.join(self.full_images_directory, file_name)
        ann_path = os.path.join(self.full_annotations_directory, file_name[:-4] + '.txt')

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img_tensor = self.transform(img)

        objects = []
        try:
            with open(ann_path, 'r') as f:
                for row in f.readlines():
                    objects.append(row.split())
        except FileNotFoundError:
            logging.warning(f"File {ann_path} not found.")
            pass

        boxes = []
        labels = []
        for obj in objects:
            # YOLO format: [class_id, x_center, y_center, width, height] (all normalized)
            class_id, cx, cy, bw, bh = map(float, obj)
            xmin = (cx - bw / 2) * w
            ymin = (cy - bh / 2) * h
            xmax = (cx + bw / 2) * w
            ymax = (cy + bh / 2) * h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(class_id))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return img_tensor, target
