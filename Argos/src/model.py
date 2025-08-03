import torch
import torchvision
from torchvision import ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from collections import OrderedDict


def get_model(number_of_output_classes,  checkpoint_path = None):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
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
    return accuracy
