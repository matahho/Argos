"""pytorchexample: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision
from torch.utils.data import Subset
import json

from collections import OrderedDict

import torchvision
from torchvision import ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from tqdm import tqdm
import os
import torch.optim as optim
from PIL import Image

from collections import OrderedDict
import json
import logging
import os
from functools import cache
import random

from tqdm import tqdm

import torch
import torchvision
from torchvision import ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import transforms




def get_model( checkpoint_path = None):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,76)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded model weights from: {checkpoint_path}")

    return model


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)




def collate_fn(batch):
    # batch is a list of (image_tensor, target_dict)
    images, targets = zip(*batch)
    return list(images), list(targets)


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    full_dataset = MTSDDataset(root_dir="./data")


    client_1_train = Subset(full_dataset, range(1 ,50))
    client_1_test = Subset(full_dataset, range(51 ,80))
    
    client_2_train = Subset(full_dataset, range(101 ,150))
    client_2_test = Subset(full_dataset, range(151 ,181))
    
    if partition_id == 1 :
        train_set = client_1_train
        val_set = client_1_test
    else :
        train_set = client_2_train
        val_set = client_2_test

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


def train(net, trainloader, valloader, epochs, learning_rate, device):
    net.to(device)
    net.train()

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    training_loss_history = []

    for epoch in range(epochs):
        total_train_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch [{epoch + 1}/{epochs}] - Training", leave=False)

        for batch_idx, (images, targets) in enumerate(pbar):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Skip samples with no ground-truth boxes
            if any(t['boxes'].numel() == 0 for t in targets):
                continue

            loss_dict = net(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(trainloader)
        training_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{epochs} — Train Loss: {avg_train_loss:.4f}")

    val_accuracy = test(net, valloader, device)
    results = {
        "val_loss": training_loss_history[-1] if training_loss_history else 0.0,
        "val_accuracy": val_accuracy,
    }
    return results





def test(net, testloader, device):
    net.eval()
    net.to(device)

    iou_threshold = 0.5
    correct_detections = 0
    total_targets = 0

    with torch.no_grad():
        pbar = tqdm(testloader, desc="Evaluating", leave=False)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Skip samples with no ground-truth boxes
            if any(t['boxes'].numel() == 0 for t in targets):
                continue

            # Get predictions
            outputs = net(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"]
                gt_boxes = target["boxes"]

                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue

                ious = ops.box_iou(pred_boxes, gt_boxes)
                max_iou_per_gt, _ = ious.max(dim=0)
                correct_detections += (max_iou_per_gt > iou_threshold).sum().item()
                total_targets += len(gt_boxes)

    accuracy = correct_detections / total_targets if total_targets > 0 else 0.0
    return 0.0 , accuracy


