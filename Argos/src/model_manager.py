from collections import OrderedDict

import torchvision
from torchvision import ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from tqdm import tqdm
import os
import torch.optim as optim
from src.utils import device_allocation


class ModelManager:
    def __init__(self, number_of_output_classes):
        self.number_of_output_classes = number_of_output_classes
        self.device = device_allocation()


    def get_model(self , checkpoint_path = None):
        """
        Returns a Faster R-CNN model with the specified number of output classes.

        Args:
            checkpoint_path (str, optional): Path to load pretrained weights (optional).

        Returns:
            model (torch.nn.Module): Faster R-CNN model ready for training.
        """

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features,self.number_of_output_classes)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded model weights from: {checkpoint_path}")

        return model

    def get_weights(self , model):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]


    def set_weights(self, model, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)



    def training(self, model, train_loader, validation_loader, epochs, learning_rate):
        model.to(self.device)
        training_loss_history = []

        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] - Training", leave=False)

            for images, targets in pbar:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                print("here1")
                loss_dict = model(images, targets)
                print("here2")
                loss = sum(loss for loss in loss_dict.values())
                print("here3")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("here4")

                total_train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            avg_loss = total_train_loss / len(train_loader)
            training_loss_history.append(avg_loss)

        validation_accuracy = self.test(model=model, test_loader=validation_loader)

        results = {
            "validation_accuracy": validation_accuracy,
            "training_loss_history": training_loss_history,
        }
        return results

    def test(self, model, test_loader):
        model.eval()
        model.to(self.device)

        iou_threshold = 0.5
        correct_detections = 0
        total_targets = 0

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating", leave=False)
            for images, targets in pbar:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

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
