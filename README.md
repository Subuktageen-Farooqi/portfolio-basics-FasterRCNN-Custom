# portfolio-basics-FasterRCNN-Custom

Object detection assignment package with:

1. **`custom_rcnn_tf.py`**: preserved TensorFlow/Keras tutorial code (transcribed from PDF).
2. **`custom_rcnn_pytorch.py`**: PyTorch version of the same educational flow.
3. **`assignment_pipeline.py`**: practical PyTorch-only implementation for assignment tasks.

## Quick usage

### Task 1 — Change number of layers
```bash
python assignment_pipeline.py task1 --num-blocks 4 --filters 64 128 256 512 --dropout 0.1
```

### Task 2 — Random testing on user image folder
```bash
python assignment_pipeline.py infer-folder --output-dir runs/inference --threshold 0.5
```
You will be prompted for an image folder path. The script processes all supported images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) and saves:
- `predictions.json`
- `summary.json`
- annotated images in `output_predictions/`

### Task 3 — Train/test custom model on labeled data
```bash
python assignment_pipeline.py train --output-dir runs/training --epochs 5 --batch-size 2 --lr 5e-4
```
You will be prompted for:
- dataset root path
- images subfolder
- COCO annotation JSON path (relative to root)

Outputs include:
- `best_model.pt`
- `training_summary.json`
- `test_predictions.json`

## Notes

- Practical training/inference uses `torchvision` Faster R-CNN (`fasterrcnn_resnet50_fpn_v2`) with official pretrained weights.
- Bounding box convention in saved predictions is always pixel coordinates: `[xmin, ymin, xmax, ymax]`.
