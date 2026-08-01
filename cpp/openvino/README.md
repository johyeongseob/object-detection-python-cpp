# OpenVINO C++ Backend

This directory contains the OpenVINO C++ implementation targeting the Intel
Arc B390 integrated GPU on Ubuntu 24.04 under WSL 2.

## Goal

Compare three person-detection runtimes under aligned COCO val2017 settings:

```text
Python + PyTorch/Ultralytics + CPU
C++ + ONNX Runtime + CPU
C++ + OpenVINO + Intel iGPU
```

The OpenVINO implementation will reuse the same YOLO11n ONNX model, image
size, confidence threshold, NMS threshold, maximum detections, warm-up count,
preprocessing, postprocessing, and COCOeval implementation.

## Environment

- WSL 2
- Ubuntu 24.04
- Intel Arc B390 integrated GPU
- OpenVINO C++ Runtime
- Intel OpenCL/Level Zero compute runtime

Initialize the OpenVINO environment in every new Ubuntu 24.04 terminal before
configuring, building, or running the C++ programs:

```bash
source /opt/intel/openvino_2026/setupvars.sh
```

This command applies the OpenVINO CMake and shared-library paths to the current
shell. It does not reinstall OpenVINO.

GPU availability must first be verified through the Intel compute runtime:

```bash
clinfo -l
```

Then configure, build, and run the OpenVINO device query:

```bash
source /opt/intel/openvino_2026/setupvars.sh
cmake -S cpp/openvino -B cpp/build-openvino -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build-openvino -j
./cpp/build-openvino/list_openvino_devices
```

The output must list an OpenVINO `GPU` device before a benchmark is accepted.

## Single-image Detection

The default command runs YOLO11n person detection on COCO image `139` using
the OpenVINO IR model and Intel iGPU:

```bash
./cpp/build-openvino/detect_person_image_openvino
```

Default model and output paths:

```text
models/yolo11n/openvino/yolo11n.xml
outputs/yolo11n/images/openvino_gpu_139_person.jpg
```

The image, output, and model paths can be overridden in that order:

```bash
./cpp/build-openvino/detect_person_image_openvino \
  datasets/coco/val2017/000000000139.jpg \
  outputs/yolo11n/images/openvino_gpu_139_person.jpg \
  models/yolo11n/openvino/yolo11n.xml
```

## Files

```text
cpp/openvino/
|-- CMakeLists.txt
|-- include/
|   `-- openvino_person_detector.hpp
|-- src/
|   |-- list_openvino_devices.cpp
|   |-- openvino_person_detector.cpp
|   |-- detect_person_image_openvino.cpp
|   `-- evaluate_coco_person_openvino.cpp
`-- README.md
```

The device query and single-image detector are implemented. The full COCO
evaluator is planned after the single-image output is validated.

OpenVINO build artifacts will be generated separately from the ONNX Runtime
build:

```text
cpp/build/             ONNX Runtime CPU build
cpp/build-openvino/    OpenVINO iGPU build
```

## Planned Artifacts

```text
outputs/yolo11n/openvino_gpu_coco_val2017_person_predictions.json
results/yolo11n/openvino_gpu_coco_person_metrics.json
```

Accuracy will be calculated by the shared tools under `evaluation/`.
