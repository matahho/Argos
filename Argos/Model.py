import logging

import torchvision
from torchvision import ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from tqdm import tqdm

from Argos.utils import device_allocation

device = device_allocation()

def get_model(num_classes, checkpoint_path=None):
    """
    Returns a Faster R-CNN model with the specified number of output classes.

    Args:
        num_classes (int): Number of object classes (including background).
        checkpoint_path (str, optional): Path to load pretrained weights (optional).

    Returns:
        model (torch.nn.Module): Faster R-CNN model ready for training.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one for our custom classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded model weights from: {checkpoint_path}")

    return model



def train(model, dataloader, optimizer, device=device):
    """
    Trains Faster R-CNN model for one epoch.

    Args:
        model (torch.nn.Module): The Faster R-CNN model.
        dataloader (DataLoader): A PyTorch DataLoader returning (images, targets).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., SGD).
        device (torch.device): CUDA or CPU.

    Returns:
        avg_loss (float): Average loss over the epoch.
    """
    model.train()
    model.to(device)
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())

    avg_loss = total_loss / num_batches
    return avg_loss




def evaluate(model, dataloader, device=device, iou_threshold=0.5):
    """
    Evaluate a Faster R-CNN model on a dataset.

    Args:
        model (torch.nn.Module): Trained Faster R-CNN model.
        dataloader (DataLoader): Validation/test dataloader.
        device (torch.device): "cuda" or "cpu".
        iou_threshold (float): IoU threshold for counting correct detections.

    Returns:
        avg_loss (float): Average loss on dataset.
        accuracy (float): Detection accuracy based on IoU.
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    num_batches = len(dataloader)
    correct_detections = 0
    total_targets = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes']
                pred_labels = output['labels']
                gt_boxes = target['boxes']
                gt_labels = target['labels']

                total_targets += len(gt_boxes)

                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue

                ious = ops.box_iou(pred_boxes, gt_boxes)

                max_iou_per_gt, matched_preds = ious.max(dim=0)
                correct = (max_iou_per_gt > iou_threshold).sum().item()
                correct_detections += correct

    avg_loss = total_loss / num_batches
    accuracy = correct_detections / total_targets if total_targets > 0 else 0.0

    return avg_loss, accuracy
