import json
import logging
import os
from collections import defaultdict
import random
from functools import cache

import torch
from PIL import Image
from torch.utils.data import Subset
from torchvision.transforms import transforms
from tqdm import tqdm

from Argos.utils import NUMBER_OF_CLIENTS

dataset_path = "../data/"
classes_json_file = os.path.join(dataset_path, "classes.json")


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
            transform=transforms.Compose([transforms.ToTensor()])
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

        #print(img_path)

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
        # print(objects)

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
@cache
def partition_dataset(dataset, num_clients):
    label_to_indices = defaultdict(list)

    for idx in tqdm(range(len(dataset))):
        _, target = dataset[idx]
        labels = target['labels'].unique().tolist()
        for label in labels:
            label_to_indices[label].append(idx)

    label_ids = list(label_to_indices.keys())
    random.shuffle(label_ids)

    client_data = defaultdict(list)
    for client_id in range(num_clients):
        assigned_labels = label_ids[client_id::num_clients]
        for lbl in assigned_labels:
            client_data[client_id].extend(label_to_indices[lbl])

    return client_data


def get_dataset_for_client(
        partition_id,
        full_dataset,
        partitioned_dataset_indices:dict,
        train_percentage=0.8, val_percentage=0.1, test_percentage=0.1
) -> [Subset , Subset , Subset]:

    assert abs(train_percentage + val_percentage + test_percentage - 1.0) < 1e-4, \
        "Splits must sum to 1.0"

    assert partition_id in list(partitioned_dataset_indices.keys()) , "Client id must be in partitioned dataset keys"

    client_samples = partitioned_dataset_indices[partition_id]
    random.shuffle(client_samples)

    total = len(client_samples)
    train_end = int(total * train_percentage)
    val_end = train_end + int(total * val_percentage)

    train_ids = client_samples[:train_end]
    val_ids = client_samples[train_end:val_end]
    test_ids = client_samples[val_end:]

    train_set = Subset(full_dataset, train_ids)
    val_set = Subset(full_dataset, val_ids)
    test_set = Subset(full_dataset, test_ids)

    return train_set, val_set, test_set


label_mapping = extract_label_mapping(classes_json_file)
number_of_classes = len(label_mapping)
dataset = MTSDDataset(root_dir=dataset_path)
dataset = Subset(dataset, list(range(100)))
partitioned_dataset_indices = partition_dataset(dataset=dataset, num_clients=NUMBER_OF_CLIENTS)
