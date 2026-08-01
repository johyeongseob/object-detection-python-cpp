# C++ Person Detection

This directory is reserved for the C++17 and CMake implementation of the
Python person-detection baseline. The implementation is the next project
milestone and is not complete yet.

## Planned Stack

- C++17
- CMake
- OpenCV C++ with the DNN module
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

## Planned Layout

```text
cpp/
|-- CMakeLists.txt
|-- include/
|   `-- person_detector.hpp
|-- src/
|   |-- main.cpp
|   `-- person_detector.cpp
`-- README.md
```

Build products will be placed in `cpp/build/` and excluded from Git.

## Planned Build and Run Commands

These commands will become valid after the C++ source and `CMakeLists.txt` are
implemented:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
./cpp/build/person_detector
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
configs/yolo11/person_detection.yaml
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
