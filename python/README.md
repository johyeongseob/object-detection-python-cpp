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

## Configuration

Defaults are read from
[`../configs/yolo11n/person_detection.yaml`](../configs/yolo11n/person_detection.yaml).
The shared configuration layout is documented in the root
[README](../README.md#configuration).

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
