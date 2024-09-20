from typing import Dict, Literal
import os
import random
import numpy as np
import torch
from loguru import logger


def set_random_seed(seed: int):
    """
    Fix numpy and torch random seed generation.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_device(params: Dict):
    """
    Set proper device for torch.
    """
    assert torch.cuda.is_available(), "No available GPU device."

    if params.get("multi_gpu_train", False):
        params["gpu_id"] = list(range(torch.cuda.device_count()))
        device = torch.device("cuda")  # use all available GPUs
    else:
        params["gpu_id"] = [0]
        device = torch.device("cuda")
    logger.info("***Use {} GPU for training.***".format(params["gpu_id"]))
    return device


def make_folder(key: str, params: Dict):
    """Make the prediction or training directories.

    Args:
        key (Text): Folder descriptor.
        params (Dict): Parameters dictionary.

    Raises:
        ValueError: Error if key is not defined.
    """
    # Make the prediction directory if it does not exist.
    if params[key] is not None:
        if not os.path.exists(params[key]):
            os.makedirs(params[key])
    else:
        raise ValueError(key + " must be defined.")

    return params[key]
