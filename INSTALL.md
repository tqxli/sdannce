# SDANNCE Installation Guide
To start with, clone the SDANNCE repository to your local machine, if not already:
```
git clone https://github.com/tqxli/sdannce.git
cd sdannce
```

## Environment Setup
Prepare the runtime environment using Conda:
```
conda create sdannce python=3.7

conda activate sdannce

conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=11.1 cudnn ffmpeg -c pytorch -c nvidia

pip install setuptools=59.5.0

pip install -e .
```
For PyTorch & Cuda compatibility on specific GPU devices, please refer to the official PyTorch instructions.