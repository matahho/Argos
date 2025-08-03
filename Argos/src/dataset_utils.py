import json
import logging
import os
from collections import defaultdict
import random
from functools import cache
import torch
from PIL import Image
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import transforms
from src.settings import DATASET_PATH, NUMBER_OF_CLIENTS, TRAIN_PERCENTAGE, VAL_PERCENTAGE, TEST_PERCENTAGE, \
    CLIENT_PARTITION_DIR, CLASSES_JSON_FILE

from tqdm import tqdm

def collate_fn(batch):
    # batch is a list of (image_tensor, target_dict)
    images, targets = zip(*batch)
    return list(images), list(targets)

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

    logging.info(f"Starting Partitioning dataset into {num_clients} clients.")

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

    logging.info(f" Partitioning dataset into {num_clients} clients finished.")

    return client_data

def get_dataset_for_client(
        partition_id,
        number_of_paritions,
        batch_size,
) -> [Subset , Subset , Subset]:
    train_percentage = TRAIN_PERCENTAGE
    val_percentage = VAL_PERCENTAGE
    test_percentage = TEST_PERCENTAGE

    assert abs(train_percentage + val_percentage + test_percentage - 1.0) < 1e-4, \
        "Splits must sum to 1.0"

    assert partition_id <= NUMBER_OF_CLIENTS, f"Partition id must be less than or equal number of clients, partition id is {partition_id}"

    partitioned_dataset_indicis = load_partitioned_indices(load_dir=CLIENT_PARTITION_DIR, num_clients=NUMBER_OF_CLIENTS)
    full_dataset = MTSDDataset(root_dir=DATASET_PATH)
    client_indices = partitioned_dataset_indicis.get(partition_id, [])

    client_samples = Subset(full_dataset, client_indices)


    total = len(client_indices)
    train_end = int(total * train_percentage)
    val_end = train_end + int(total * val_percentage)

    train_ids = client_indices[:train_end]
    val_ids = client_indices[train_end:val_end]
    test_ids = client_indices[val_end:]

    train_set = Subset(client_samples, train_ids)
    val_set = Subset(client_samples, val_ids)
    test_set = Subset(client_samples, test_ids)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


def save_partitioned_indices(partitioned_dataset: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for client_id, indices in partitioned_dataset.items():
        client_file = os.path.join(save_dir, f"client_{client_id}.json")
        with open(client_file, 'w') as f:
            json.dump(indices, f)
    logging.info(f"Partitioned indices saved to {save_dir}")

def load_partitioned_indices(load_dir: str, num_clients: int) -> dict:
    partitioned_dataset_indicis = {}
    for client_id in range(num_clients):
        client_file = os.path.join(load_dir, f"client_{client_id}.json")
        if os.path.exists(client_file):
            with open(client_file, 'r') as f:
                partitioned_dataset_indicis[client_id] = json.load(f)
        else:
            raise FileNotFoundError(f"Partition file for client {client_id} not found in {load_dir}")
    logging.info(f"Partitioned indices loaded from {load_dir}")
    return partitioned_dataset_indicis


# NOTE : to save the partitioned dataset

# dataset = MTSDDataset(DATASET_PATH)
# dataset = Subset(dataset , range(100))
# partitioned_dataset_indices = partition_dataset(dataset, num_clients=NUMBER_OF_CLIENTS)
# save_partitioned_indices(partitioned_dataset_indices, CLIENT_PARTITION_DIR)
