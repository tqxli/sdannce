import os
from typing import Dict
import torch

import dannce.config as config
import dannce.engine.run.inference as inference
import dannce.engine.models.posegcn.nets as sdanncenets
from dannce.engine.models.nets import (
    initialize_train,
    initialize_prediction,
    initialize_com_train,
)
from dannce.engine.trainer import *
from dannce.engine.run.run_utils import *


def _train_prep(params: Dict, model_type: str = "dannce"):
    """Prepare for training by setting up experiment, dataset, and model.

    Args:
        params (Dict): Parameters dictionary.
        model_type (str): dannce OR sdannce.
    """
    logger, device = experiment_setup(params, "dannce_train")
    (
        params,
        base_params,
        shared_args,
        shared_args_train,
        shared_args_valid,
    ) = config.setup_train(params)

    spec_args = params["dataset_args"]
    spec_args = {} if spec_args is None else spec_args

    # Prepare dataset
    dataset_preparer = set_dataset(params)
    train_dataloader, valid_dataloader, n_cams = dataset_preparer(
        params,
        base_params,
        shared_args,
        shared_args_train,
        shared_args_valid,
        **spec_args
    )

    if model_type == "sdannce":
        sdannce_model_params = params["graph_cfg"]
        params["use_features"] = sdannce_model_params.get("use_features", False)

    # Build network
    logger.info("Initializing Network...")
    model, optimizer, lr_scheduler = initialize_train(
        params, n_cams, device, model_type
    )
    logger.info(model)
    logger.success("Ready for training!\n")

    return (
        device,
        params,
        train_dataloader,
        valid_dataloader,
        model,
        optimizer,
        lr_scheduler,
    )


def _predict_prep(params: Dict, model_type: str = "dannce"):
    """Prepare for prediction by setting up experiment and model.

    Args:
        params (Dict): Parameters dictionary.
    """
    logger, device = experiment_setup(params, "dannce_predict")
    params, valid_params = config.setup_predict(params)

    checkpoint_params = torch.load(params["dannce_predict_model"])["params"]
    if "graph_cfg" in checkpoint_params:
        sdannce_model_params = checkpoint_params["graph_cfg"]
    elif "custom_model" in checkpoint_params:
        sdannce_model_params = checkpoint_params["custom_model"]
    else:
        sdannce_model_params = {}

    if model_type == "sdannce":
        params["use_features"] = sdannce_model_params.get("use_features", False)

    (predict_generator, _, camnames, partition,) = make_dataset_inference(
        params, valid_params
    )

    model = initialize_prediction(params, len(camnames[0]), device, model_type)
    return device, params, sdannce_model_params, partition, predict_generator, model


def dannce_train(params: Dict):
    """Train DANNCE network.

    Args:
        params (Dict): Parameters dictionary.

    Raises:
        Exception: Error if training mode is invalid.
    """
    (
        device,
        params,
        train_dataloader,
        valid_dataloader,
        model,
        optimizer,
        lr_scheduler,
    ) = _train_prep(params, "dannce")

    # set up trainer
    trainer_class = DannceTrainer
    trainer = trainer_class(
        params=params,
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        device=device,
        logger=logger,
        visualize_batch=False,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


def dannce_predict(params: Dict):
    """Predict with DANNCE network

    Args:
        params (Dict): Paremeters dictionary.
    """
    device, params, _, partition, predict_generator, model = _predict_prep(
        params, "dannce"
    )

    inference.infer_sdannce(predict_generator, params, {}, model, partition, device)


def sdannce_train(params: Dict):
    """Train SDANNCE network.

    Args:
        params (Dict): Parameters dictionary.

    Raises:
        Exception: Error if training mode is invalid.
    """
    (
        device,
        params,
        train_dataloader,
        valid_dataloader,
        model,
        optimizer,
        lr_scheduler,
    ) = _train_prep(params, "sdannce")

    # set up trainer
    sdannce_model_params = params["graph_cfg"]
    trainer = GCNTrainer(
        params=params,
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        device=device,
        logger=logger,
        visualize_batch=False,
        lr_scheduler=lr_scheduler,
        predict_diff=sdannce_model_params.get("predict_diff", True),
        relpose=sdannce_model_params.get("relpose", True),
    )

    trainer.train()


def sdannce_predict(params: Dict):
    """Predict with SDANNCE network

    Args:
        params (Dict): Paremeters dictionary.
    """
    (
        device,
        params,
        custom_model_params,
        partition,
        predict_generator,
        model,
    ) = _predict_prep(params, "sdannce")

    # inference
    inference.infer_sdannce(
        predict_generator, params, custom_model_params, model, partition, device
    )


def com_train(params: Dict):
    """Train COM network
    Args:
        params (Dict): Parameters dictionary.
    """
    logger, device = experiment_setup(params, "com_train")
    params, train_params, valid_params = config.setup_com_train(params)
    train_dataloader, valid_dataloader = make_data_com(
        params, train_params, valid_params,
    )

    # Build network
    logger.info("Initializing Network...")
    model, optimizer, lr_scheduler = initialize_com_train(params, device)
    logger.info(model)
    logger.success("Ready for training!\n")

    # set up trainer
    trainer_class = COMTrainer
    trainer = trainer_class(
        params=params,
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        device=device,
        logger=logger,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


def com_predict(params: Dict):
    """Predict with COM network
    """
    logger, device = experiment_setup(params, "com_predict")
    params, predict_params = config.setup_com_predict(params)
    (
        predict_generator,
        params,
        partition,
        camera_mats,
        cameras,
        datadict,
    ) = make_dataset_com_inference(params, predict_params)

    logger.info("Initializing Network...")
    model = initialize_com_train(params, device)[0]
    model.load_state_dict(torch.load(params["com_predict_weights"])["state_dict"])
    model.eval()

    # do inference frame by frame
    save_data = {}
    endIdx = (
        np.min(
            [
                params["start_sample"] + params["max_num_samples"],
                len(predict_generator),
            ]
        )
        if params["max_num_samples"] != "max"
        else len(predict_generator)
    )

    save_data = inference.infer_com(
        params["start_sample"],
        endIdx,
        predict_generator,
        params,
        model,
        partition,
        save_data,
        camera_mats,
        cameras,
        device,
    )

    filename = (
        "com3d"
        if params["max_num_samples"] == "max"
        else "com3d%d" % (params["start_sample"])
    )
    processing.save_COM_checkpoint(
        save_data,
        params["com_predict_dir"],
        datadict,
        cameras,
        params,
        file_name=filename,
    )
