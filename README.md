# Object Detection: Python to C++

This project explores an end-to-end object detection workflow: running and
evaluating a model on COCO val2017 with Python and reproducing the inference
pipeline with C++ and CMake.

## Project structure

```text
detection/
├── configs/     # Reproducible model and experiment settings
├── python/      # Python inference and evaluation
├── cpp/         # C++ inference and CMake project
├── models/      # Model-specific weight directories (not tracked by Git)
├── datasets/    # Shared datasets (not tracked by Git)
├── outputs/     # Generated predictions and images (not tracked by Git)
└── results/     # Small evaluation summaries tracked by Git
```

The Python and C++ implementations share the datasets stored under
`datasets/`.

## Development environment

This project uses Ubuntu 22.04 through WSL 2 on Windows. Python inference,
evaluation, and C++ builds all run in the same Linux environment.

### Open the project in WSL

Open the Windows project directory from WSL with:

```bash
cd /mnt/c/Users/YOUR_WINDOWS_USERNAME/Desktop/detection
```

The `/mnt/c` mount makes the project easy to access from both Windows and WSL.
It may be slower than the native WSL filesystem when installing packages or
reading many small files.

### Install system packages

```bash
sudo apt update
sudo apt install -y \
  python3-venv \
  python3-pip \
  build-essential \
  cmake \
  pkg-config \
  libopencv-dev \
  git \
  unzip
```

### Create and activate the Python environment

Create the virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify that the environment is active:

```bash
which python
python --version
python -m pip --version
```

The expected Python path is:

```text
/mnt/c/Users/YOUR_WINDOWS_USERNAME/Desktop/detection/.venv/bin/python
```

Reactivate the environment from the project root whenever a new WSL terminal
is opened:

```bash
source .venv/bin/activate
```

Deactivate the environment with:

```bash
deactivate
```

### Install CPU-only PyTorch

The current environment does not use an NVIDIA GPU, so install the CPU-only
PyTorch wheels:

```bash
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

`CUDA: False` is the expected result for this CPU-only environment.

### Install the Python dependencies

Install the pinned dependencies from the project root:

```bash
python -m pip install -r python/requirements.txt
```

### Download the YOLO11n weights

Model weights are not tracked by Git. After installing the Python
dependencies, download the official Ultralytics YOLO11n weights into the
configured model directory:

```bash
mkdir -p models/yolo11n
cd models/yolo11n
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
cd ../..
```

Verify that the model exists from the project root:

```bash
ls -lh models/yolo11n/yolo11n.pt
```

The expected project-relative model path is:

```text
models/yolo11n/yolo11n.pt
```

### Run Python person detection

The default YOLO11 person-detection settings are stored in
`configs/yolo11/person_detection.yaml`.

```bash
python python/detect_person.py
```

Command-line arguments override YAML values for one-off experiments:

```bash
python python/detect_person.py --image-id 139 --confidence 0.50
```

### Evaluate one image

Compare person predictions with COCO ground truth using IoU matching:

```bash
python python/evaluate_person_image.py --image-id 139
```

### Evaluate COCO val2017

Run official COCO metrics for the person category on all 5,000 validation
images:

```bash
python python/evaluate_coco_person.py
```

Run a small end-to-end smoke test before the full CPU evaluation:

```bash
python python/evaluate_coco_person.py --limit 10
```

The raw per-image predictions are written to `outputs/yolo11n/`, and generated
images are written to `outputs/yolo11n/images/`. A compact metrics and
environment summary is written to `results/yolo11n/coco_person_metrics.json`
and can be committed to Git.

## Python Baseline Results

YOLO11n was evaluated for the `person` category on all 5,000 COCO val2017
images. Raw predictions used a low confidence threshold so that COCOeval could
construct the full precision-recall curve.

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

### Latency and Throughput

| Metric | Result |
| --- | ---: |
| Mean model inference | 29.50 ms |
| P50 model inference | 28.25 ms |
| P95 model inference | 40.89 ms |
| End-to-end throughput | 22.94 images/s |
| Total prediction time | 217.93 s |

Model inference timing excludes most image I/O and Python orchestration. The
end-to-end throughput includes image loading, preprocessing, inference,
postprocessing, and conversion to COCO prediction records.

### Environment

| Component | Value |
| --- | --- |
| CPU | Intel Core Ultra X7 358H |
| Device | CPU |
| Operating environment | WSL 2, Ubuntu 22.04 |
| Python | 3.10.12 |
| PyTorch | 2.13.0+cpu |
| Ultralytics | 8.4.114 |

### Evaluation Configuration

| Setting | Value |
| --- | ---: |
| Model | YOLO11n |
| Input size | 640 |
| Batch size | 1 |
| Evaluated images | 5,000 |
| Category | person |
| Evaluation confidence threshold | 0.001 |
| NMS IoU threshold | 0.70 |
| Maximum detections per image | 100 |

The complete machine-readable result is available in
[`results/yolo11n/coco_person_metrics.json`](results/yolo11n/coco_person_metrics.json).

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
