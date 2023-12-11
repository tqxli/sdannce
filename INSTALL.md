# SDANNCE Installation Guide
To start with, clone the SDANNCE repository to your local machine, if not already:
```
git clone https://github.com/tqxli/sdannce.git
cd sdannce
```

## Environment Setup
Prepare the runtime environment using Conda:
```
conda create --name sdannce python=3.8

conda activate sdannce

# REPLACEABLE with other workable PyTorch installation
conda install pytorch=1.9.1 torchvision=0.10.1 cudatoolkit=11.1 cudnn ffmpeg -c pytorch -c nvidia

pip install setuptools==59.5.0

pip install -e .
```

The installation was tested on Linux (Ubuntu 20.04) with a NVIDIA GPU (RTX 3090, A5000, A6000) and Conda version=4.10.3.

The code was tested for Python >= 3.7, PyTorch >= 1.8 (and torchvision that matches the PyTorch installation according to the official instructions https://pytorch.org/get-started/previous-versions/). 

Users should be mindful of choosing PyTorch versions compatible with their local CUDA installation. 