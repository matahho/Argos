import torch
import torchvision
from torchvision import ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from collections import OrderedDict


def get_model(number_of_output_classes,  checkpoint_path = None):

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,number_of_output_classes)

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


def train(net, trainloader, valloader, epochs, learning_rate, device):
    net.to(device)
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
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

    # Get validation loss and accuracy separately
    val_loss = validate(net, valloader, device)
    val_accuracy = test(net, valloader, device)

    results = {
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }
    return results


def test(net, testloader, device):
    net.eval()
    net.to(device)
    iou_threshold = 0.1
    correct_detections = 0
    total_targets = 0

    # Save original score threshold and temporarily lower it
    original_score_thresh = net.roi_heads.score_thresh
    net.roi_heads.score_thresh = 0.0  # Get ALL predictions, regardless of confidence

    with torch.no_grad():
        pbar = tqdm(testloader, desc="Evaluating", leave=False)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Skip samples with no ground-truth boxes
            if any(t['boxes'].numel() == 0 for t in targets):
                continue
            # Get predictions (with all confidence scores)
            outputs = net(images)
            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"]
                gt_boxes = target["boxes"]
                if len(gt_boxes) == 0:  # Only skip if ground truth is empty
                    continue
                # Handle case where model predicts no boxes
                if len(pred_boxes) == 0:
                    total_targets += len(gt_boxes)
                    continue
                ious = ops.box_iou(pred_boxes, gt_boxes)
                max_iou_per_gt, _ = ious.max(dim=0)
                correct_detections += (max_iou_per_gt > iou_threshold).sum().item()
                total_targets += len(gt_boxes)

    # Restore original score threshold
    net.roi_heads.score_thresh = original_score_thresh

    accuracy = correct_detections / total_targets if total_targets > 0 else 0.0
    return accuracy


def validate(net, valloader, device):
    """Calculate validation loss only."""
    # Save current mode
    was_training = net.training
    # Set to training mode to calculate loss
    net.train()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in valloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Skip samples with no ground-truth boxes
            valid_targets = [t for t in targets if t['boxes'].numel() > 0]
            if not valid_targets:
                continue
            loss_dict = net(images, valid_targets)
            # This should be a dictionary (not a list)
            loss = sum(loss for loss in loss_dict.values())
            total_loss += loss.item()
            num_batches += 1

    # Restore original mode
    net.train(was_training)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss