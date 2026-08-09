#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
venv_dir="${project_root}/.venv"
python_bin="${venv_dir}/bin/python"
model_dir="${project_root}/models/yolo11n"
pt_model="${model_dir}/yolo11n.pt"
onnx_model="${model_dir}/yolo11n.onnx"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required." >&2
    exit 1
fi

if [[ ! -x "${python_bin}" ]]; then
    echo "Creating Python virtual environment: ${venv_dir}"
    python3 -m venv "${venv_dir}"
fi

if ! "${python_bin}" -c "import ultralytics, onnx" >/dev/null 2>&1; then
    echo "Installing model export dependencies..."
    "${python_bin}" -m pip install \
        --requirement "${project_root}/python/requirements.txt"
fi

mkdir -p "${model_dir}"

echo "Downloading YOLO11n PyTorch weights when not already available..."
PROJECT_PT_MODEL="${pt_model}" "${python_bin}" -c \
    "import os; from ultralytics import YOLO; YOLO(os.environ['PROJECT_PT_MODEL'])"

echo "Exporting YOLO11n to ONNX..."
PROJECT_PT_MODEL="${pt_model}" "${python_bin}" -c \
    "import os; from ultralytics import YOLO; YOLO(os.environ['PROJECT_PT_MODEL']).export(format='onnx', imgsz=640, opset=12, simplify=False, dynamic=False, nms=False, device='cpu')"

PROJECT_ONNX_MODEL="${onnx_model}" "${python_bin}" -c \
    "import os, onnx; path = os.environ['PROJECT_ONNX_MODEL']; model = onnx.load(path); onnx.checker.check_model(model); print(f'ONNX model ready: {path}')"
