import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class BackboneConfig:
    num_blocks: int = 3
    filters: Tuple[int, ...] = (64, 128, 256)
    dropout: float = 0.0


class ConfigurableBackbone(nn.Module):
    """Task 1: configurable educational backbone for layer-count experiments."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        if config.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if len(config.filters) < config.num_blocks:
            raise ValueError("filters length must be >= num_blocks")

        layers: List[nn.Module] = []
        in_channels = 3
        for block_idx in range(config.num_blocks):
            out_channels = config.filters[block_idx]
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            if config.dropout > 0:
                layers.append(nn.Dropout2d(p=config.dropout))
            in_channels = out_channels

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class COCODetectionDataset(Dataset):
    """Minimal COCO JSON reader without pycocotools."""

    def __init__(self, images_dir: Path, annotation_file: Path):
        if not images_dir.exists() or not images_dir.is_dir():
            raise FileNotFoundError(f"images_dir does not exist: {images_dir}")
        if not annotation_file.exists() or not annotation_file.is_file():
            raise FileNotFoundError(f"annotation_file does not exist: {annotation_file}")

        with annotation_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        if not images or not annotations or not categories:
            raise ValueError("COCO JSON must include non-empty 'images', 'annotations', and 'categories'.")

        self.images_dir = images_dir
        self.id_to_image = {img["id"]: img for img in images}

        cat_ids = sorted(cat["id"] for cat in categories)
        self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}
        self.label_to_name = {
            self.cat_id_to_label[cat["id"]]: cat["name"]
            for cat in categories
            if cat["id"] in self.cat_id_to_label
        }

        anns_by_image: Dict[int, List[dict]] = {}
        for ann in annotations:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.samples: List[Tuple[Path, List[dict], int]] = []
        for image_id, img in self.id_to_image.items():
            file_name = img["file_name"]
            image_path = images_dir / file_name
            image_anns = anns_by_image.get(image_id, [])
            if not image_path.exists() or len(image_anns) == 0:
                continue
            self.samples.append((image_path, image_anns, image_id))

        if not self.samples:
            raise ValueError("No usable image/annotation pairs found. Check image paths and annotations.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, anns, image_id = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            xmin = float(x)
            ymin = float(y)
            xmax = float(x + w)
            ymax = float(y + h)
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.cat_id_to_label[ann["category_id"]])
            areas.append(float(w * h))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if not boxes:
            raise ValueError(f"Image {image_path} has no valid boxes.")

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        return F.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_detection_model(num_classes: int, device: torch.device) -> nn.Module:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    return model


def gather_images(folder: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        candidates = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    else:
        candidates = [p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(candidates)


def run_folder_inference(output_dir: Path, score_threshold: float = 0.5):
    user_input = input("Enter image folder path for batch inference: ").strip()
    input_dir = Path(user_input).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Invalid folder path: {input_dir}")

    image_paths = gather_images(input_dir, recursive=True)
    if not image_paths:
        raise ValueError(
            f"No supported images found in {input_dir}. Supported: {sorted(IMAGE_EXTS)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    categories = weights.meta.get("categories", [])

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "output_predictions"
    annotated_dir.mkdir(exist_ok=True)

    records = []
    total_with_dets = 0

    with torch.inference_mode():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            image_tensor = preprocess(image).to(device)
            prediction = model([image_tensor])[0]

            boxes = prediction["boxes"].detach().cpu()
            labels = prediction["labels"].detach().cpu()
            scores = prediction["scores"].detach().cpu()

            preds = []
            draw = ImageDraw.Draw(image)

            for box, label, score in zip(boxes, labels, scores):
                score_v = float(score.item())
                if score_v < score_threshold:
                    continue
                box_v = [float(v) for v in box.tolist()]  # pixel coords [xmin, ymin, xmax, ymax]
                label_idx = int(label.item())
                label_name = categories[label_idx] if label_idx < len(categories) else str(label_idx)

                preds.append({"label": label_name, "score": score_v, "box": box_v})
                draw.rectangle(box_v, outline="red", width=2)
                draw.text((box_v[0], box_v[1]), f"{label_name}:{score_v:.2f}", fill="red")

            if preds:
                total_with_dets += 1

            image.save(annotated_dir / image_path.name)

            records.append(
                {
                    "image": str(image_path.name),
                    "width": width,
                    "height": height,
                    "predictions": preds,
                }
            )

    predictions_json = output_dir / "predictions.json"
    with predictions_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    summary = {
        "total_images": len(image_paths),
        "total_images_with_detections": total_with_dets,
        "threshold": score_threshold,
        "model": "fasterrcnn_resnet50_fpn_v2",
        "date_time_utc": datetime.now(timezone.utc).isoformat(),
        "coordinates": "pixel [xmin, ymin, xmax, ymax]",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved predictions to: {predictions_json}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")
    print(f"Annotated images in: {annotated_dir}")


def run_training_pipeline(output_dir: Path, epochs: int = 2, batch_size: int = 2, lr: float = 5e-4):
    print("Provide dataset paths for training (COCO JSON format).")
    dataset_root = Path(input("Dataset root folder path: ").strip()).expanduser().resolve()
    images_subdir = input("Images subfolder (example: images): ").strip()
    ann_rel_path = input("Annotation JSON path relative to root (example: annotations/train.json): ").strip()

    images_dir = dataset_root / images_subdir
    annotation_file = dataset_root / ann_rel_path

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    dataset = COCODetectionDataset(images_dir=images_dir, annotation_file=annotation_file)

    n_total = len(dataset)
    if n_total < 5:
        raise ValueError("Need at least 5 labeled images for a train/val/test split.")

    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n_train - 1)

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    num_classes = len(dataset.label_to_name) + 1  # + background
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_detection_model(num_classes=num_classes, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += float(loss.item())

        model.train()
        val_loss_total = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]
                loss_dict = model(images, targets)
                val_loss_total += float(sum(loss_dict.values()).item())

        avg_train_loss = train_loss_total / max(1, len(train_loader))
        avg_val_loss = val_loss_total / max(1, len(val_loader))
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.pt"
    torch.save(model.state_dict(), model_path)

    test_predictions = []
    with torch.inference_mode():
        for images, targets in test_loader:
            image = images[0].to(device)
            pred = model([image])[0]
            tgt = targets[0]
            image_id = int(tgt["image_id"].item())
            pred_boxes = pred["boxes"].detach().cpu().tolist()
            pred_labels = pred["labels"].detach().cpu().tolist()
            pred_scores = pred["scores"].detach().cpu().tolist()

            mapped_preds = []
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                mapped_preds.append(
                    {
                        "label": dataset.label_to_name.get(int(label), str(label)),
                        "score": float(score),
                        "box": [float(v) for v in box],
                    }
                )

            test_predictions.append({"image_id": image_id, "predictions": mapped_preds})

    with (output_dir / "test_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(test_predictions, f, indent=2)

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split_sizes": {"train": n_train, "val": n_val, "test": n_test},
                "history": history,
                "best_val_loss": best_val_loss,
                "model_path": str(model_path),
                "test_predictions_file": str(output_dir / "test_predictions.json"),
                "coordinates": "pixel [xmin, ymin, xmax, ymax]",
            },
            f,
            indent=2,
        )

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved test predictions to: {output_dir / 'test_predictions.json'}")


def show_task1_demo(args):
    config = BackboneConfig(
        num_blocks=args.num_blocks,
        filters=tuple(args.filters),
        dropout=args.dropout,
    )
    model = ConfigurableBackbone(config)
    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    print(model)
    print(f"Task-1 demo output shape: {tuple(output.shape)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Assignment pipeline for Faster R-CNN custom tasks.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_task1 = sub.add_parser("task1", help="Task 1: change the number of layers.")
    p_task1.add_argument("--num-blocks", type=int, default=3)
    p_task1.add_argument("--filters", type=int, nargs="+", default=[64, 128, 256, 512])
    p_task1.add_argument("--dropout", type=float, default=0.0)

    p_infer = sub.add_parser("infer-folder", help="Task 2: run inference on all images in a user-provided folder.")
    p_infer.add_argument("--output-dir", type=Path, default=Path("runs/inference"))
    p_infer.add_argument("--threshold", type=float, default=0.5)

    p_train = sub.add_parser("train", help="Task 3: train/test custom model with labeled COCO-format data.")
    p_train.add_argument("--output-dir", type=Path, default=Path("runs/training"))
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--lr", type=float, default=5e-4)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "task1":
        show_task1_demo(args)
    elif args.command == "infer-folder":
        run_folder_inference(output_dir=args.output_dir, score_threshold=args.threshold)
    elif args.command == "train":
        run_training_pipeline(
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
