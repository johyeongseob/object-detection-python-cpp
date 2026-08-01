# Person Detection: Python to C++

An end-to-end person-detection project using YOLO11n and COCO val2017. The
same person-detection task is evaluated with Python/PyTorch on CPU, C++/ONNX
Runtime on CPU, and C++/OpenVINO on an Intel integrated GPU.

Language-specific instructions:

- [Python setup and usage](python/README.md)
- [C++ dependencies and ONNX Runtime usage](cpp/README.md)
- [OpenVINO C++ and Intel iGPU usage](cpp/openvino/README.md)

## YOLO11n Runtime Comparison

All implementations evaluated the COCO val2017 `person` category over 5,000
images with an input size of 640, batch size 1, confidence threshold 0.001,
NMS IoU threshold 0.70, and at most 100 detections per image.

The delta column is calculated as `OpenVINO iGPU - Python CPU`. Negative
latency and wall-time deltas are improvements.

| Metric | Python CPU | C++ CPU | OpenVINO iGPU | OpenVINO delta vs Python |
| --- | ---: | ---: | ---: | ---: |
| AP50-95 | 0.5292 | 0.5338 | 0.5338 | +0.0046 |
| AP50 | 0.7679 | 0.7732 | 0.7728 | +0.0049 |
| AP75 | 0.5659 | 0.5699 | 0.5695 | +0.0036 |
| AR100 | 0.6476 | 0.6488 | 0.6488 | +0.0012 |
| Predictions | 129,074 | 128,230 | 128,052 | -1,022 |
| Mean model inference | 29.50 ms | 23.01 ms | 5.79 ms | -23.71 ms (-80.4%, 5.10x faster) |
| P50 model inference | 28.25 ms | 19.16 ms | 5.75 ms | -22.50 ms (-79.7%, 4.92x faster) |
| P95 model inference | 40.89 ms | 43.32 ms | 6.08 ms | -34.80 ms (-85.1%, 6.72x faster) |
| End-to-end throughput | 22.94 images/s | 20.39 images/s | 47.69 images/s | +24.75 images/s (+107.9%, 2.08x) |
| Total prediction time | 217.93 s | 245.24 s | 104.84 s | -113.09 s (-51.9%, 2.08x faster) |

OpenVINO retained accuracy parity with ONNX Runtime while reducing mean model
inference latency by approximately 74.8% relative to C++ CPU inference. The
complete machine-readable comparison is available in
[`results/yolo11n/python_cpp_openvino_comparison.json`](results/yolo11n/python_cpp_openvino_comparison.json).

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
|-- evaluation/              Shared COCO evaluation and comparison tools
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

The Python and ONNX Runtime CPU pipelines use WSL 2 with Ubuntu 22.04. The
OpenVINO iGPU pipeline uses Ubuntu 24.04 for Intel Arc B390 compute-runtime
support. Clone the repository and enter it from WSL:

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
models/yolo11n/yolo11n.onnx    ONNX Runtime C++ model
models/yolo11n/openvino/       OpenVINO IR model and metadata
```

## Configuration and Artifacts

Shared settings are stored in
[`configs/yolo11/person_detection.yaml`](configs/yolo11/person_detection.yaml).
Generated files follow this layout:

```text
outputs/yolo11n/               Large, reproducible artifacts ignored by Git
results/yolo11n/               Compact metric summaries tracked by Git
```

The Python, C++ CPU, and OpenVINO iGPU evaluations use the same dataset, input
size, thresholds, batch size, and COCO evaluator wherever possible.

## Key Finding

Object scale remains the clearest accuracy weakness: the Python baseline falls
from `0.7779` AP for large people to `0.2858` AP for small people. OpenVINO
iGPU preserves the C++ CPU accuracy (`0.5338` AP50-95) while improving mean
model inference from `23.01 ms` to `5.79 ms` and end-to-end throughput from
`20.39` to `47.69` images/s.

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
