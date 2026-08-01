# Person Detection: Python to C++

An end-to-end person-detection project using YOLO11n and COCO val2017. The
Python baseline covers inference, COCO evaluation, and CPU benchmarking. The
next phase reproduces the pipeline in C++17 with CMake.

## Status

- [x] Inspect COCO val2017 annotations
- [x] Run YOLO11n person detection in Python
- [x] Evaluate the `person` category on all 5,000 validation images
- [x] Record CPU accuracy, latency, and throughput
- [ ] Implement equivalent C++ inference with CMake
- [ ] Evaluate all 5,000 images in C++
- [ ] Compare Python and C++ accuracy and performance

Language-specific instructions:

- [Python setup and usage](python/README.md)
- [C++ dependencies, build, and roadmap](cpp/README.md)

## Python Baseline

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
[`results/yolo11n/coco_person_metrics.json`](results/yolo11n/coco_person_metrics.json).

## Pipeline

```text
COCO image
  -> preprocess
  -> YOLO11n inference
  -> person filtering and NMS
  -> COCO-format predictions
  -> COCOeval metrics and runtime summary
```

## Repository Layout

```text
.
|-- configs/                 Shared experiment settings
|-- python/                  Python programs, dependencies, and documentation
|-- cpp/                     C++17, CMake, and C++ documentation
|-- datasets/                Downloaded data and dataset utilities
|-- models/                  Models shared by Python and C++
|-- outputs/                 Generated images and raw predictions
|-- results/                 Compact evaluation summaries tracked by Git
|-- .gitignore
`-- README.md
```

Datasets, model weights, and generated outputs are excluded from Git. Python
and C++ share the same `configs/`, `datasets/`, `models/`, and `outputs/`
directories to support fair comparisons.

## Common Data and Model Setup

The project is developed in WSL 2 with Ubuntu 22.04. Clone the repository and
enter it from WSL:

```bash
git clone https://github.com/johyeongseob/object-detection-python-cpp.git
cd object-detection-python-cpp
```

For a repository stored on the Windows filesystem:

```bash
cd /mnt/c/Users/YOUR_WINDOWS_USERNAME/PATH_TO_PROJECT
```

Download COCO val2017 images and annotations:

```bash
sudo apt update
sudo apt install -y wget unzip
mkdir -p datasets/coco
wget -c -P datasets/coco \
  https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip
wget -c -P datasets/coco \
  https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q datasets/coco/val2017.zip -d datasets/coco
unzip -q datasets/coco/annotations_trainval2017.zip -d datasets/coco
```

### Download YOLO11n

YOLO11n is a common project asset shared by Python and C++. Model files are
ignored by Git. After installing the Python dependencies described in
[`python/README.md`](python/README.md), download the official Ultralytics
weights from the repository root:

```bash
mkdir -p models/yolo11n
(
  cd models/yolo11n
  python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
)
ls -lh models/yolo11n/yolo11n.pt
```

The shared model locations are:

```text
models/yolo11n/yolo11n.pt      Python model
models/yolo11n/yolo11n.onnx    C++ deployment model (planned)
```

The ONNX artifact will be added when the C++ pipeline is implemented.

## Configuration and Artifacts

Shared settings are stored in
[`configs/yolo11/person_detection.yaml`](configs/yolo11/person_detection.yaml).
Generated files follow this layout:

```text
outputs/yolo11n/               Large, reproducible artifacts ignored by Git
results/yolo11n/               Compact metric summaries tracked by Git
```

The Python and C++ evaluations will use the same dataset, input size,
thresholds, batch size, and COCO evaluator wherever possible.

## Key Finding

Object scale is the clearest weakness in the Python baseline. AP falls from
`0.7779` for large people to `0.2858` for small people. The next milestone is
to reproduce the pipeline in C++ and measure accuracy parity and runtime.

## Citation

This project uses Ultralytics YOLO11:

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
