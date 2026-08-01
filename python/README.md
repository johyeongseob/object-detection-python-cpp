# Python Person Detection

This directory contains the completed Python baseline for YOLO11n person
detection and COCO val2017 evaluation. Run all commands from the repository
root.

## Requirements

- WSL 2 with Ubuntu 22.04
- Python 3.10
- `python3-venv` and `python3-pip`
- CPU PyTorch, Ultralytics, pycocotools, OpenCV, and PyYAML

Install the system packages:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

Create the project-local environment and install the pinned Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

Verify the environment:

```bash
python -c "import torch, torchvision, ultralytics, pycocotools; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('ultralytics:', ultralytics.__version__); print('CUDA:', torch.cuda.is_available()); print('pycocotools: OK')"
```

`CUDA: False` is expected for the documented CPU baseline.


## Programs

### Inspect COCO annotations

```bash
python python/inspect_coco.py --image-id 139
```

This prints image metadata and ground-truth objects. COCO bounding boxes use
the `[x, y, width, height]` format.

### Run person detection on one image

```bash
python python/detect_person.py --image-id 139
```

Override the visualization confidence for one run:

```bash
python python/detect_person.py --image-id 139 --confidence 0.50
```

The annotated image is saved under `outputs/yolo11n/images/`.

### Evaluate one image

```bash
python python/evaluate_person_image.py --image-id 139
```

This converts boxes from `xywh` to `xyxy`, performs one-to-one greedy IoU
matching, and reports TP, FP, FN, precision, and recall.

### Smoke-test COCO evaluation

```bash
python -u python/evaluate_coco_person.py --limit 10
```

Limited-run artifacts use an image-count suffix and do not overwrite the full
evaluation results.

### Evaluate all 5,000 images

```bash
python -u python/evaluate_coco_person.py
```

Images are processed one at a time to keep memory usage predictable. The
evaluation confidence is `0.001`, allowing COCOeval to construct a
precision-recall curve.

## Evaluation Results

YOLO11n was evaluated on all 5,000 COCO val2017 images. Only the `person`
category was scored, while images without people were retained so false
positives remained part of the evaluation.

### Accuracy

| Metric | Result |
| --- | ---: |
| AP50-95 | 0.5292 |
| AP50 | 0.7679 |
| AP75 | 0.5659 |
| AP small | 0.2858 |
| AP medium | 0.6172 |
| AP large | 0.7779 |
| AR100 | 0.6476 |

### Runtime Performance

| Metric | Result |
| --- | ---: |
| Mean model inference | 29.50 ms |
| P50 model inference | 28.25 ms |
| P95 model inference | 40.89 ms |
| End-to-end throughput | 22.94 images/s |
| Total prediction time | 217.93 s |

Inference alone averaged approximately 33.9 FPS. End-to-end throughput also
includes image loading, preprocessing, postprocessing, and COCO conversion.

### Evaluation Environment

| Setting | Value |
| --- | --- |
| Model | YOLO11n |
| Dataset | COCO val2017, 5,000 images |
| Category | person |
| Input size | 640 |
| Batch size | 1 |
| Evaluation confidence | 0.001 |
| NMS IoU threshold | 0.70 |
| Maximum detections | 100 per image |
| Device | CPU |
| CPU | Intel Core Ultra X7 358H |
| Environment | WSL 2, Ubuntu 22.04 |
| Python | 3.10.12 |
| PyTorch | 2.13.0+cpu |
| Ultralytics | 8.4.114 |

The complete machine-readable result is available in
[`../results/yolo11n/python_coco_person_metrics.json`](../results/yolo11n/python_coco_person_metrics.json).

## Configuration

Defaults are read from
[`../configs/yolo11/person_detection.yaml`](../configs/yolo11/person_detection.yaml).
The file defines the model and dataset paths, device, input size, confidence
thresholds, NMS threshold, and output directories.

Supported command-line arguments override YAML settings:

```text
CLI option > YAML configuration
```

The thresholds have distinct purposes:

- Detection confidence `0.25`: readable single-image visualization
- Evaluation confidence `0.001`: COCO precision-recall evaluation
- NMS IoU `0.70`: suppression among model predictions
- Match IoU `0.50`: single-image prediction-to-ground-truth matching

## Output

```text
outputs/yolo11n/
|-- images/                                      Annotated images
`-- python_coco_val2017_person_predictions.json  Raw predictions

results/yolo11n/
`-- python_coco_person_metrics.json              Metrics and environment
```

`outputs/` is ignored by Git. Compact summaries under `results/` are tracked.
The root [README](../README.md) contains the project-wide runtime comparison.
