# SDANNCE Installation Guide
## SDANNCE
Clone the SDANNCE repository.
```
git clone
cd sdannce
```

## Environment Setup
Prepare the runtime environment (conda):

`conda create sdannce python=3.7`

`conda activate sdannce`

`conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=11.1 cudnn ffmpeg -c pytorch -c nvidia`

`pip install setuptools=59.5.0`

`pip install -e .`

For PyTorch & cuda compatibility, please refer to the official PyTorch instructions.