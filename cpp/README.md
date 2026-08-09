# C++ Person Detection

This directory contains the C++17 and CMake implementation of YOLO11n person
detection with ONNX Runtime.

See [`openvino/README.md`](openvino/README.md) for the Intel iGPU
implementation.

## Technology Stack

- C++17
- CMake
- ONNX Runtime C++ 1.23.2
- OpenCV C++ for image processing and visualization
- cpp-httplib for the HTTP server
- YOLO11n exported to ONNX
- WSL 2 with Ubuntu 22.04

YOLO11n is exported from PyTorch to ONNX and executed in C++ with ONNX
Runtime.

## System Dependencies

Install the C++ system dependencies in Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev
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

## Model Preparation

The C++ programs do not download model files automatically. Prepare the
YOLO11n ONNX model from the repository root:

```bash
bash scripts/prepare_yolo11n.sh
```

The generated model is stored at:

```text
models/yolo11n/yolo11n.onnx
```

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
|   |-- person_detection_server.cpp
|   `-- verify_yolo11n_onnx.cpp
`-- README.md
```


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
