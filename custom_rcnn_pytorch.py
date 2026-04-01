import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import numpy as np


# Step 1: Define the Backbone Network (Simple CNN)
class Backbone(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)


# Step 2: Build the Region Proposal Network (RPN)
class RPN(nn.Module):
    def __init__(self, in_channels=512, num_anchors=9):
        super().__init__()
        self.rpn_conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.rpn_class = nn.Conv2d(512, num_anchors * 2, kernel_size=1)
        self.rpn_bbox = nn.Conv2d(512, num_anchors * 4, kernel_size=1)

    def forward(self, feature_map):
        x = F.relu(self.rpn_conv(feature_map))
        rpn_class = torch.sigmoid(self.rpn_class(x))
        rpn_bbox = self.rpn_bbox(x)
        return rpn_class, rpn_bbox


# Step 3: Implement the ROI Pooling Layer
class ROIPoolingLayer(nn.Module):
    def __init__(self, pool_size=(7, 7), spatial_scale=1.0):
        super().__init__()
        self.pool_size = pool_size
        self.spatial_scale = spatial_scale

    def forward(self, feature_map, rois):
        # rois format: [N, 5] => [batch_idx, x1, y1, x2, y2]
        return ops.roi_align(
            feature_map,
            rois,
            output_size=self.pool_size,
            spatial_scale=self.spatial_scale,
            aligned=True,
        )


# Step 4: Create Classification and Regression Heads
class Heads(nn.Module):
    def __init__(self, in_channels=512, pool_size=(7, 7), num_classes=80):
        super().__init__()
        flattened = in_channels * pool_size[0] * pool_size[1]
        self.fc = nn.Linear(flattened, 1024)
        self.classifier = nn.Linear(1024, num_classes)
        self.bbox_regressor = nn.Linear(1024, num_classes * 4)

    def forward(self, pooled_rois):
        x = pooled_rois.flatten(start_dim=1)
        x = F.relu(self.fc(x))
        class_logits = self.classifier(x)
        bbox_reg = self.bbox_regressor(x)
        class_probs = F.softmax(class_logits, dim=1)
        return class_probs, bbox_reg


# Step 5: Assemble the Complete Faster R-CNN Model
class FasterRCNNToy(nn.Module):
    def __init__(self, num_classes=80, pool_size=(7, 7)):
        super().__init__()
        self.backbone = Backbone()
        self.rpn = RPN(in_channels=512)
        self.roi_pool = ROIPoolingLayer(pool_size=pool_size, spatial_scale=1 / 8)
        self.heads = Heads(in_channels=512, pool_size=pool_size, num_classes=num_classes)

    def forward(self, images, rois):
        feature_map = self.backbone(images)
        rpn_class, rpn_bbox = self.rpn(feature_map)
        pooled_rois = self.roi_pool(feature_map, rois)
        classifier, bbox_regressor = self.heads(pooled_rois)
        return rpn_class, rpn_bbox, classifier, bbox_regressor


# Step 6: Calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2

    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou


# Step 7: Calculate Precision and Recall
def calculate_precision_recall(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    tp, fp = 0, 0
    matched_gt = set()

    for i, pred_box in enumerate(pred_boxes):
        if pred_scores[i] < iou_threshold:
            continue

        best_iou = 0
        best_gt_index = -1

        for j, gt_box in enumerate(gt_boxes):
            if pred_labels[i] == gt_labels[j] and j not in matched_gt:
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = j

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_index)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall


# Step 8: Calculate Average Precision (AP)
def calculate_ap(precisions, recalls):
    precisions = [0.0] + precisions + [0.0]
    recalls = [0.0] + recalls + [1.0]

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = [i for i in range(1, len(recalls)) if recalls[i] != recalls[i - 1]]
    ap = sum((recalls[i] - recalls[i - 1]) * precisions[i] for i in indices)
    return ap


# Step 9: Calculate Mean Average Precision (mAP) Across All Classes
def calculate_map(predictions, ground_truths, num_classes, iou_threshold=0.5):
    average_precisions = []

    for class_id in range(1, num_classes):
        all_precisions = []
        all_recalls = []

        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = [box for i, box in enumerate(pred['boxes']) if pred['labels'][i] == class_id]
            pred_scores = [score for i, score in enumerate(pred['scores']) if pred['labels'][i] == class_id]
            gt_boxes = [box for i, box in enumerate(gt['boxes']) if gt['labels'][i] == class_id]

            precision, recall = calculate_precision_recall(
                pred_boxes,
                pred_scores,
                [class_id] * len(pred_boxes),
                gt_boxes,
                [class_id] * len(gt_boxes),
                iou_threshold,
            )

            all_precisions.append(precision)
            all_recalls.append(recall)

        ap = calculate_ap(all_precisions, all_recalls)
        average_precisions.append(ap)

    mAP = float(np.mean(average_precisions))
    return mAP


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 80
    model = FasterRCNNToy(num_classes=num_classes).to(device)
    print(model)

    # Example forward pass with synthetic data
    images = torch.randn(1, 3, 224, 224, device=device)
    rois = torch.tensor(
        [
            [0, 10.0, 20.0, 100.0, 140.0],
            [0, 50.0, 60.0, 180.0, 200.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        outputs = model(images, rois)
    print("Forward pass complete. Output tensor shapes:", [o.shape for o in outputs])

    # Placeholder predictions and ground truths for testing
    predictions = [
        {'boxes': [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.8, 0.8]], 'scores': [0.9, 0.7], 'labels': [1, 2]}
    ]
    ground_truths = [
        {'boxes': [[0.12, 0.12, 0.42, 0.42], [0.5, 0.5, 0.8, 0.8]], 'labels': [1, 2]}
    ]

    mean_ap = calculate_map(predictions, ground_truths, num_classes)
    print(f"Mean Average Precision (mAP): {mean_ap:.2f}")
