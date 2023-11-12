"""Handle training and prediction for DANNCE and COM networks."""
import os
from typing import Dict
import psutil
import torch

import dannce.config as config
import dannce.engine.run.inference as inference
import dannce.engine.models.posegcn.nets as sdanncenets
from dannce.engine.models.nets import (
    initialize_train,
    initialize_model,
    initialize_com_train,
)
from dannce.engine.trainer import *
from dannce.engine.run.run_utils import *

# from loguru import logger

process = psutil.Process(os.getpid())


def dannce_train(params: Dict):
    """Train DANNCE network.

    Args:
        params (Dict): Parameters dictionary.

    Raises:
        Exception: Error if training mode is invalid.
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

    dataset_preparer = set_dataset(params)
    train_dataloader, valid_dataloader, n_cams = dataset_preparer(
        params,
        base_params,
        shared_args,
        shared_args_train,
        shared_args_valid,
        logger,
        **spec_args
    )

    # Build network
    logger.info("Initializing Network...")
    model, optimizer, lr_scheduler = initialize_train(params, n_cams, device, logger)
    logger.info(model)
    logger.success("Ready for training!\n")

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
    logger, device = experiment_setup(params, "dannce_predict")
    params, valid_params = config.setup_predict(params)
    if params["dataset"] == "rat7m":
        predict_generator = dataset.RAT7MNPYDataset(train=False)
        predict_generator_sil = None
        camnames = [["Camera1", "Camera2", "Camera3", "Camera4", "Camera5", "Camera6"]]
        partition = {"valid_sampleIDs": np.arange(len(predict_generator))}
    else:
        (
            predict_generator,
            predict_generator_sil,
            camnames,
            partition,
        ) = make_dataset_inference(params, valid_params)
        
    logger.info("Initializing Network...")
    model = initialize_model(params, len(camnames[0]), device)

    # load predict model
    if params.get("inference_ttt", None) is not None:
        ttt_params = params["inference_ttt"]
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model_params, lr=params["lr"], eps=1e-7)
        save_data = inference.inference_ttt(
            predict_generator,
            params,
            model,
            optimizer,
            device,
            partition,
            online=ttt_params.get("online", False),
            niter=ttt_params.get("niter", 20),
            transform=ttt_params.get("transform", False),
            downsample=ttt_params.get("downsample", 1),
        )
        inference.save_results(params, save_data)
        return

    model.load_state_dict(torch.load(params["dannce_predict_model"])["state_dict"])
    model.eval()

    save_data = inference.infer_dannce(
        predict_generator,
        params,
        model,
        partition,
        device,
        params["n_markers"],
        predict_generator_sil,
        save_heatmaps=False,
    )
    inference.save_results(params, save_data)


def sdannce_train(params: Dict):
    """Train SDANNCE network.

    Args:
        params (Dict): Parameters dictionary.

    Raises:
        Exception: Error if training mode is invalid.
    """
    logger, device = experiment_setup(params, "dannce_train")
    (
        params,
        base_params,
        shared_args,
        shared_args_train,
        shared_args_valid,
    ) = config.setup_train(params)

    # handle specific params
    custom_model_params = params["graph_cfg"]

    spec_args = params["dataset_args"]
    spec_args = {} if spec_args is None else spec_args

    dataset_preparer = set_dataset(params)
    train_dataloader, valid_dataloader, n_cams = dataset_preparer(
        params,
        base_params,
        shared_args,
        shared_args_train,
        shared_args_valid,
        logger,
        **spec_args
    )

    # Build network
    logger.info("Initializing Network...")

    params["use_features"] = custom_model_params.get("use_features", False)
    pose_generator = initialize_train(params, n_cams, "cpu", logger)[0]

    # second stage: pose refiner
    model_class = getattr(sdanncenets, custom_model_params.get("model", "PoseGCN"))
    model = model_class(
        params,
        custom_model_params,
        pose_generator,
    )

    # load full-model checkpoint (if exists)
    if "checkpoint" in custom_model_params.keys():
        model.load_state_dict(
            torch.load(custom_model_params["checkpoint"])["state_dict"]
        )
    model = model.to(device)
    logger.info(model)

    # optimization
    optimizer = set_optimizer(params, model)
    lr_scheduler = set_lr_scheduler(params, optimizer, logger)

    logger.success("Ready for training!\n")

    # set up trainer
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
        predict_diff=custom_model_params.get("predict_diff", True),
        relpose=custom_model_params.get("relpose", True),
    )

    trainer.train()


def sdannce_predict(params):
    """Predict with SDANNCE network

    Args:
        params (Dict): Paremeters dictionary.
    """
    logger, device = experiment_setup(params, "dannce_predict")

    params, valid_params = config.setup_predict(params)

    # handle specific params
    # load in params saved in checkpoint to ensure consistency
    try:
        custom_model_params = torch.load(params["dannce_predict_model"])["params"][
            "graph_cfg"
        ]
    except:
        custom_model_params = torch.load(params["dannce_predict_model"])["params"][
            "custom_model"
        ]

    predict_generator, _, camnames, partition = make_dataset_inference(
        params, valid_params
    )

    logger.info("Initializing Network...")
    # first stage: pose generator
    params["use_features"] = custom_model_params.get("use_features", False)
    pose_generator = initialize_model(params, len(camnames[0]), "cpu")

    # second stage: pose refiner
    model_class = getattr(sdanncenets, custom_model_params.get("model", "PoseGCN"))
    model = model_class(
        params,
        custom_model_params,
        pose_generator,
    ).to(device)

    # (DEV) test-time optimization
    if params.get("inference_ttt", None) is not None:
        ttt_params = params["inference_ttt"]
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model_params, lr=params["lr"], eps=1e-7)
        save_data = inference.inference_ttt(
            predict_generator,
            params,
            model,
            optimizer,
            device,
            partition,
            online=ttt_params.get("online", False),
            niter=ttt_params.get("niter", 20),
            transform=ttt_params.get("transform", False),
            downsample=ttt_params.get("downsample", 1),
            gcn=True,
        )
        inference.save_results(params, save_data)
        return

    # load predict model
    model.load_state_dict(torch.load(params["dannce_predict_model"])["state_dict"])
    model.eval()

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
        params, train_params, valid_params, logger
    )

    # Build network
    logger.info("Initializing Network...")
    model, optimizer, lr_scheduler = initialize_com_train(params, device, logger)
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


def com_predict(params):
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
    model = initialize_com_train(params, device, logger)[0]
    model.load_state_dict(torch.load(params["com_predict_weights"])["state_dict"])
    model.eval()

    # do frame-wise inference
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
