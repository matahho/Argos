import torchvision
from torchvision import ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from tqdm import tqdm
import os
from Argos.settings import DEVICE

def get_model(num_classes, checkpoint_path=None, device=DEVICE):
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

    return model.to(device)



def train(model, dataloader, optimizer, epoch, checkpoint_dir=None, device=DEVICE):
    """
    Trains Faster R-CNN model for one epoch and optionally saves a checkpoint.

    Args:
        model (torch.nn.Module): The Faster R-CNN model.
        dataloader (DataLoader): PyTorch DataLoader returning (images, targets).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., SGD).
        epoch (int): Current epoch number (used in checkpoint filename).
        checkpoint_dir (str, optional): Directory to save model checkpoints.
        device (torch.device): Device to train on ("cuda" or "cpu").

    Returns:
        avg_loss (float): Average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", batch=f"{batch_idx+1}/{num_batches}")

    avg_loss = total_loss / num_batches

    # Save checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"fasterrcnn_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"[✓] Saved checkpoint to {checkpoint_path}")

    return avg_loss


def evaluate(model, dataloader, device=DEVICE, iou_threshold=0.5):
    """
    Evaluate a Faster R-CNN model on a dataset.

    Computes only IoU-based accuracy (no loss).

    Args:
        model (torch.nn.Module): Trained Faster R-CNN model.
        dataloader (DataLoader): Validation/test DataLoader.
        device (torch.device): "cuda" or "cpu".
        iou_threshold (float): IoU threshold for counting correct detections.

    Returns:
        accuracy (float): Detection accuracy based on IoU.
    """
    model.eval()
    model.to(device)

    correct_detections = 0
    total_targets = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # skip samples with no ground-truth boxes
            if any(t['boxes'].numel() == 0 for t in targets):
                continue

            # get predictions only
            outputs = model(images)

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
    return accuracy
