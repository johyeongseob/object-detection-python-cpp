# Person Detection: Python to C++

An end-to-end person detection project built around YOLO11n and COCO val2017.
The Python baseline covers inference, COCO evaluation, and CPU benchmarking.
The next phase will reproduce the same pipeline in C++ with CMake.

## Status

- [x] Inspect COCO val2017 annotations
- [x] Run YOLO11n person detection in Python
- [x] Compare predictions with ground truth using IoU matching
- [x] Evaluate the `person` category on all 5,000 validation images
- [x] Record CPU latency, throughput, and environment metadata
- [ ] Analyze and document representative failure cases
- [ ] Add focused tests for box conversion, IoU, and matching
- [ ] Implement the equivalent C++ inference pipeline
- [ ] Compare Python and C++ accuracy and performance

## Python Baseline

YOLO11n was evaluated on all 5,000 COCO val2017 images. Only the `person`
category was scored, but images without people were retained so that false
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

The model averaged approximately 33.9 FPS for inference alone. End-to-end
throughput also includes image loading, preprocessing, postprocessing, and
conversion to COCO prediction records.

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

The complete machine-readable result is available at
[`results/yolo11n/coco_person_metrics.json`](results/yolo11n/coco_person_metrics.json).

## Pipeline

```text
COCO image
  -> resize and preprocess
  -> YOLO11n inference
  -> person-class filtering
  -> confidence filtering and NMS
  -> COCO-format predictions
  -> COCOeval metrics and runtime summary
```

## Repository Layout

```text
.
|-- configs/                 Reproducible model and evaluation settings
|-- python/                  Python inference and evaluation programs
|-- cpp/                     Planned C++ and CMake implementation
|-- datasets/                Downloaded data and dataset utilities
|-- models/                  Downloaded model weights
|-- outputs/                 Generated images and raw predictions
|-- results/                 Small evaluation summaries tracked by Git
|-- .gitignore
`-- README.md
```

Downloaded datasets, model weights, and generated outputs are intentionally
excluded from Git. Python and C++ will share the same `datasets/`, `models/`,
and `outputs/` structure.

## Setup

The project is tested in Ubuntu 22.04 through WSL 2.

### 1. Install System Packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git wget unzip
```

### 2. Clone the Repository

```bash
git clone https://github.com/johyeongseob/object-detection-python-cpp.git
cd object-detection-python-cpp
```

When working from an existing Windows checkout, open it from WSL instead:

```bash
cd /mnt/c/Users/YOUR_WINDOWS_USERNAME/PATH_TO_PROJECT
```

### 3. Create the Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

Confirm that the CPU build of PyTorch is active:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

`CUDA: False` is expected for the documented CPU baseline.

## Prepare COCO val2017

Download the 5,000 validation images and the instance annotations from the
official COCO storage bucket:

```bash
mkdir -p datasets/coco
wget -c -P datasets/coco \
  https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip
wget -c -P datasets/coco \
  https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip -q datasets/coco/val2017.zip -d datasets/coco
unzip -q datasets/coco/annotations_trainval2017.zip -d datasets/coco
```

Verify the dataset:

```bash
find datasets/coco/val2017 -maxdepth 1 -type f -name '*.jpg' | wc -l
test -f datasets/coco/annotations/instances_val2017.json && echo "annotations OK"
```

Expected output:

```text
5000
annotations OK
```

An optional person-image subset can be created for quick inspection. It must
not replace the full 5,000-image set during formal evaluation because images
without people are required to measure false positives.

```bash
python datasets/coco/create_person_subset.py
```

## Download YOLO11n

Model weights are not tracked by Git. Download the official Ultralytics
YOLO11n weights after installing the Python dependencies:

```bash
mkdir -p models/yolo11n
(
  cd models/yolo11n
  python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
)
```

Verify the configured model path:

```bash
ls -lh models/yolo11n/yolo11n.pt
```

## Usage

All commands below run from the repository root with `.venv` activated.

### Inspect COCO Annotations

```bash
python python/inspect_coco.py --image-id 139
```

This prints dataset statistics and the ground-truth objects stored in COCO's
`[x, y, width, height]` box format.

### Detect People in One Image

```bash
python python/detect_person.py --image-id 139
```

Override YAML settings for a one-off experiment:

```bash
python python/detect_person.py --image-id 139 --confidence 0.50
```

The annotated image is written to `outputs/yolo11n/images/`.

### Evaluate One Image

```bash
python python/evaluate_person_image.py --image-id 139
```

This converts COCO boxes from `xywh` to `xyxy`, greedily matches predictions
to unique ground truths, and reports IoU, TP, FP, FN, precision, and recall.

### Smoke-Test the Full Evaluation Pipeline

```bash
python -u python/evaluate_coco_person.py --limit 10
```

Smoke-test artifacts use a `_10_images` suffix and do not overwrite full
evaluation results.

### Evaluate All 5,000 Images

```bash
python -u python/evaluate_coco_person.py
```

The full evaluation processes one image at a time for predictable memory use.
It uses a low confidence threshold (`0.001`) so COCOeval can construct the
precision-recall curve.

## Configuration

The default experiment is defined in
[`configs/yolo11/person_detection.yaml`](configs/yolo11/person_detection.yaml).
It contains model, device, image size, detection thresholds, dataset paths,
output paths, and evaluation settings.

Command-line values override YAML values where supported:

```text
CLI option > YAML configuration
```

The detection and evaluation confidence thresholds intentionally differ:

- `0.25` is used for readable single-image visualizations.
- `0.001` is used to collect candidates for COCO AP evaluation.

The NMS IoU threshold and ground-truth matching IoU threshold also serve
different purposes and are stored separately.

## Generated Artifacts

```text
outputs/yolo11n/
|-- images/                                      Annotated result images
`-- coco_val2017_person_predictions.json         Raw COCO predictions

results/yolo11n/
`-- coco_person_metrics.json                     Accuracy and runtime summary
```

`outputs/` is ignored by Git because these files are large or reproducible.
Compact JSON summaries under `results/` are tracked for comparison across
models, runtimes, and languages.

## Key Finding

Object scale is the clearest current failure mode. AP falls from `0.7779` for
large people to `0.2858` for small people. The next Python milestone is to
create a reproducible failure-case gallery and test the box and matching
utilities before starting the C++ implementation.

## Citation

This project uses Ultralytics YOLO11. Please cite the original software as:

```bibtex
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```
