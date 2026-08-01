# C++ Person Detection

This directory contains the C++17 and CMake implementation of the Python
person-detection baseline. The current first milestone verifies that ONNX
Runtime C++ can load and execute the exported YOLO11n ONNX model.

The next backend is OpenVINO on the Intel Arc B390 integrated GPU. Its setup
and implementation plan are documented in
[`openvino/README.md`](openvino/README.md).

## Planned Stack

- C++17
- CMake
- ONNX Runtime C++ 1.23.2
- OpenCV C++ for image processing and visualization (next milestone)
- YOLO11n exported to ONNX
- WSL 2 with Ubuntu 22.04

ONNX is used as a deployment artifact between the trained PyTorch model and
the C++ runtime. It is not a separate application layer.

## System Dependencies

C++ dependencies are installed through Ubuntu rather than
`python/requirements.txt`:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev
```

Verify the toolchain:

```bash
g++ --version
cmake --version
pkg-config --version
pkg-config --modversion opencv4
test -f /usr/include/opencv4/opencv2/dnn.hpp \
  && echo "OpenCV DNN headers: OK"
```

`CMakeLists.txt` will describe compilation and library linking. It does not
replace the operating-system package installation commands above.

Download and extract the ONNX Runtime C++ SDK from the repository root:

```bash
mkdir -p cpp/third_party
wget -c -P cpp/third_party \
  https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
tar -xzf cpp/third_party/onnxruntime-linux-x64-1.23.2.tgz \
  -C cpp/third_party
```

The downloaded `cpp/third_party/` directory is excluded from Git.

## Current Layout

```text
cpp/
|-- CMakeLists.txt
|-- include/
|   `-- person_detector.hpp
|-- openvino/
|   `-- README.md
|-- src/
|   |-- detect_person_image.cpp
|   |-- evaluate_coco_person.cpp
|   |-- person_detector.cpp
|   `-- verify_yolo11n_onnx.cpp
`-- README.md
```

Build products will be placed in `cpp/build/` and excluded from Git.

## Build and Run

Run these commands from the repository root:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
./cpp/build/verify_yolo11n_onnx
```

The program uses ONNX Runtime C++ to load `models/yolo11n/yolo11n.onnx`,
performs one forward pass with a zero-filled `[1, 3, 640, 640]` tensor, and
prints the output shape. A different model path can be passed as the first
argument:

```bash
./cpp/build/verify_yolo11n_onnx path/to/model.onnx
```

The expected YOLO11n output shape is `[1, 84, 8400]`.

Run person detection on the default COCO image:

```bash
./cpp/build/detect_person_image
```

The default output is
`outputs/yolo11n/images/cpp_139_person.jpg`. Custom image, output, and model
paths can be passed in that order:

```bash
./cpp/build/detect_person_image input.jpg output.jpg model.onnx
```

Smoke-test the C++ COCO inference loop:

```bash
./cpp/build/evaluate_coco_person --limit 10
```

Run inference on all 5,000 validation images:

```bash
./cpp/build/evaluate_coco_person
```

Ten warm-up iterations run before measurement by default. Raw COCO predictions
are written under `outputs/yolo11n/`, while the compact metrics summary is
written to `results/yolo11n/cpp_coco_person_metrics.json`. The C++ program
writes runtime fields first, and the shared COCOeval command adds AP and AR to
the same JSON file.

Evaluate the C++ predictions with the same Python COCOeval implementation used
by the Python baseline:

```bash
python evaluation/evaluate_cpp_coco_person.py
```

For a smoke test, use the same limit for inference and evaluation:

```bash
./cpp/build/evaluate_coco_person --limit 10
python evaluation/evaluate_cpp_coco_person.py --limit 10
```

## Shared Assets

Models and datasets remain at the repository root so both languages consume
the same inputs. Their preparation is documented in the root
[README](../README.md#common-data-and-model-setup).

```text
models/yolo11n/yolo11n.pt
models/yolo11n/yolo11n.onnx
datasets/coco/val2017/
datasets/coco/annotations/instances_val2017.json
configs/yolo11n/person_detection.yaml
```

The initial C++ milestone will load one image, run person-only inference, and
save `outputs/yolo11n/images/cpp_139_person.jpg`.

## Evaluation Plan

After single-image parity is confirmed, C++ will process all 5,000 COCO
validation images with batch size 1. Predictions will be evaluated by the
same Python `pycocotools.COCOeval` path used for the baseline, keeping the
evaluator constant.

The final comparison will include:

- COCO AP and AR accuracy
- Bounding-box and confidence parity
- Mean, P50, and P95 inference latency
- End-to-end throughput
- Runtime environment and configuration

See the root [README](../README.md) for the completed Python baseline.
