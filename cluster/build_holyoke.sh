#!/bin/bash
#SBATCH --job-name=sdannce_build
#SBATCH --output=sdannce_build.out
#SBATCH --partition=olveczky,shared,test
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --time=0-01:00

set -e

# Load modules
module load Anaconda3/2020.11
module load cuda/11.1.0-fasrc01 cudnn/8.0.4.30_cuda11.1-fasrc01

# Create conda environment if it doesn't exist
if [ ! -d $HOME/.conda/envs/sdannce ]
then
    conda create --name sdannce python=3.7 -Y
fi

# Activate conda environment
source activate sdannce

# Install ffmpeg
conda install ffmpeg -c conda-forge -y

# Install PyTorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install setuptools==59.5.0

# Install sdannce
pip install -e .
