"""Setup file for dannce."""
from setuptools import setup, find_packages

setup(
    name="sdannce",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "six",
        "pyyaml",
        "imageio==2.8.0",
        "imageio-ffmpeg",
        "numpy",
        "scikit-image",
        "matplotlib",
        "attr",
        "attrs",
        "multiprocess",
        "opencv-python",
        "tensorboard",
        "mat73",
        "psutil",
        "tqdm",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "dannce = dannce.cli:main",
            "dannce-predict-single-batch = cluster.multi_gpu:dannce_predict_single_batch",
            "sdannce-predict-single-batch = cluster.multi_gpu:sdannce_predict_single_batch",
            "dannce-train-single-batch = cluster.grid:dannce_train_single_batch",
            "com-predict-single-batch = cluster.multi_gpu:com_predict_single_batch",
            "dannce-inference-sbatch = cluster.multi_gpu:submit_inference",
            "dannce-inference = cluster.multi_gpu:inference",
            "sdannce-inference = cluster.multi_gpu:sdannce_inference",
        ]
    },
)
