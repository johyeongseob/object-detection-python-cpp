# Object Detection: Python to C++

This project explores an end-to-end object detection workflow: running and
evaluating a model on COCO val2017 with Python and reproducing the inference
pipeline with C++ and CMake.

## Project structure

```text
detection/
├── python/      # Python inference and evaluation
├── cpp/         # C++ inference and CMake project
└── datasets/    # Shared datasets (not tracked by Git)
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
