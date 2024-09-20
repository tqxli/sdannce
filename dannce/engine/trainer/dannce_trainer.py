import csv
import os

import imageio
import numpy as np
import torch
from tqdm import tqdm

from dannce.engine.trainer.base_trainer import BaseTrainer
from dannce.engine.trainer.train_utils import (LossHelper, MetricHelper,
                                               prepare_batch)
from dannce.engine.utils.augmentation import construct_augmented_batch
from dannce.engine.utils.image import norm_im


class DANNCETrainer(BaseTrainer):
    """
    Trainer class for DANNCE base networks.
    """

    def __init__(
        self,
        device,
        train_dataloader,
        valid_dataloader,
        lr_scheduler=None,
        visualize_batch=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.loss = LossHelper(self.params)
        self.metrics = MetricHelper(self.params)
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler

        self.visualize_batch = visualize_batch

        self.split = False  # self.params.get("social_joint_training", False)

        # whether each batch only contains transformed versions of one single instance
        self.form_batch = self.params.get("form_batch", False)
        self.form_bs = self.params.get("form_bs", None)
        self.per_batch_sample = self.params["batch_size"]

        # set up csv file for tracking training and validation stats
        stats_file = open(
            os.path.join(self.checkpoint_dir, "training.csv"), "w", newline=""
        )
        stats_writer = csv.writer(stats_file)
        self.stats_keys = [*self.loss.names, *self.metrics.names]
        self.train_stats_keys = ["train_" + k for k in self.stats_keys]
        self.valid_stats_keys = ["val_" + k for k in self.stats_keys]
        stats_writer.writerow(["Epoch", *self.train_stats_keys, *self.valid_stats_keys])
        stats_file.close()

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            # open csv
            stats_file = open(
                os.path.join(self.params["dannce_train_dir"], "training.csv"),
                "a",
                newline="",
            )
            stats_writer = csv.writer(stats_file)
            stats = [epoch]
            # train
            train_stats = self._train_epoch(epoch)

            for k in self.stats_keys:
                stats.append(train_stats[k])

            result_msg = f"Epoch[{epoch}/{self.epochs}]\n" + "".join(
                f"train_{k}: {val:.4f}\n" for k, val in train_stats.items()
            )

            # validation
            valid_stats = self._valid_epoch(epoch)

            for k in self.stats_keys:
                stats.append(valid_stats[k])

            result_msg = result_msg + "".join(
                f"val_{k}: {val:.4f}\n" for k, val in valid_stats.items()
            )
            self.logger.info(result_msg)

            # write stats to csv
            stats_writer.writerow(stats)
            stats_file.close()

            # write stats to tensorboard
            for k, v in zip(
                [*self.train_stats_keys, *self.valid_stats_keys], stats[1:]
            ):
                self.writer.add_scalar(k, v, epoch)

            # save checkpoints after each save period or at the end of training
            self._save_checkpoint(epoch)

    def _forward(self, epoch, batch, train=True):
        volumes, grid_centers, keypoints_3d_gt, aux = prepare_batch(batch, self.device)

        if self.visualize_batch:
            self.visualize(epoch, volumes)
            return

        if train and self.form_batch:
            volumes, grid_centers, aux = construct_augmented_batch(
                volumes.permute(0, 2, 3, 4, 1),
                grid_centers,
                # batch_size=self.form_bs,
                aux=aux if aux is None else aux.permute(0, 2, 3, 4, 1),
                # n_sample=self.per_batch_sample
                copies_per_sample=self.form_bs // self.per_batch_sample,
            )
            volumes = volumes.permute(0, 4, 1, 2, 3)
            aux = aux if aux is None else aux.permute(0, 4, 1, 2, 3)
            keypoints_3d_gt = keypoints_3d_gt.repeat(self.form_bs, 1, 1)

        keypoints_3d_pred, heatmaps, _ = self.model(volumes, grid_centers)

        keypoints_3d_gt, keypoints_3d_pred, heatmaps = self._split_data(
            keypoints_3d_gt, keypoints_3d_pred, heatmaps
        )

        return keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux

    def _train_epoch(self, epoch):
        self.model.train()

        # with torch.autograd.set_detect_anomaly(False):
        epoch_loss_dict, epoch_metric_dict = {}, {}
        pbar = tqdm(self.train_dataloader)
        for batch in pbar:
            self.optimizer.zero_grad()
            (
                keypoints_3d_gt,
                keypoints_3d_pred,
                heatmaps,
                grid_centers,
                aux,
            ) = self._forward(epoch, batch)

            total_loss, loss_dict = self.loss.compute_loss(
                keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux
            )
            result = f"Epoch[{epoch}/{self.epochs}] " + "".join(
                f"train_{loss}: {val:.4f} " for loss, val in loss_dict.items()
            )
            pbar.set_description(result)

            total_loss.backward()
            self.optimizer.step()

            epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

            if len(self.metrics.names) != 0:
                metric_dict = self.metrics.evaluate(
                    keypoints_3d_pred.detach().cpu().numpy(),
                    keypoints_3d_gt.clone().cpu().numpy(),
                )
                epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        epoch_loss_dict, epoch_metric_dict = (
            self._average(epoch_loss_dict),
            self._average(epoch_metric_dict),
        )
        return {**epoch_loss_dict, **epoch_metric_dict}

    def _valid_epoch(self, epoch):
        self.model.eval()

        epoch_loss_dict = {}
        epoch_metric_dict = {}

        pbar = tqdm(self.valid_dataloader)
        with torch.no_grad():
            for batch in pbar:
                (
                    keypoints_3d_gt,
                    keypoints_3d_pred,
                    heatmaps,
                    grid_centers,
                    aux,
                ) = self._forward(epoch, batch, False)

                _, loss_dict = self.loss.compute_loss(
                    keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux
                )
                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                if len(self.metrics.names) != 0:
                    metric_dict = self.metrics.evaluate(
                        keypoints_3d_pred.detach().cpu().numpy(),
                        keypoints_3d_gt.clone().cpu().numpy(),
                    )
                    epoch_metric_dict = self._update_step(
                        epoch_metric_dict, metric_dict
                    )

        epoch_loss_dict, epoch_metric_dict = (
            self._average(epoch_loss_dict),
            self._average(epoch_metric_dict),
        )
        return {**epoch_loss_dict, **epoch_metric_dict}

    def _split_data(self, keypoints_3d_gt, keypoints_3d_pred, heatmaps):
        if not self.split:
            return keypoints_3d_gt, keypoints_3d_pred, heatmaps

        keypoints_3d_gt = keypoints_3d_gt.reshape(
            *keypoints_3d_gt.shape[:2], 2, -1
        ).permute(0, 2, 1, 3)
        keypoints_3d_gt = keypoints_3d_gt.reshape(-1, *keypoints_3d_gt.shape[2:])
        keypoints_3d_pred = keypoints_3d_pred.reshape(
            *keypoints_3d_pred.shape[:2], 2, -1
        ).permute(0, 2, 1, 3)
        keypoints_3d_pred = keypoints_3d_pred.reshape(-1, *keypoints_3d_pred.shape[2:])
        heatmaps = heatmaps.reshape(heatmaps.shape[0], 2, -1, *heatmaps.shape[2:])
        heatmaps = heatmaps.reshape(-1, *heatmaps.shape[2:])

        return keypoints_3d_gt, keypoints_3d_pred, heatmaps

    def _update_step(self, epoch_dict, step_dict):
        if len(epoch_dict) == 0:
            for k, v in step_dict.items():
                epoch_dict[k] = [v]
        else:
            for k, v in step_dict.items():
                epoch_dict[k].append(v)
        return epoch_dict

    def _average(self, epoch_dict):
        for k, v in epoch_dict.items():
            valid_num = sum([item > 0 for item in v])
            epoch_dict[k] = sum(v) / valid_num if valid_num > 0 else 0.0
        return epoch_dict

    def _rewrite_csv(self):
        stats_file = open(
            os.path.join(self.params["dannce_train_dir"], "training.csv"),
            "w",
            newline="",
        )
        stats_writer = csv.writer(stats_file)
        stats_writer.writerow(["Epoch", *self.train_stats_keys, *self.valid_stats_keys])
        stats_file.close()

    def _add_loss_attr(self, names):
        self.stats_keys = names + self.stats_keys
        self.train_stats_keys = [f"train_{k}" for k in names] + self.train_stats_keys
        self.valid_stats_keys = [f"val_{k}" for k in names] + self.valid_stats_keys

        self._rewrite_csv()

    def _del_loss_attr(self, names):
        for name in names:
            self.stats_keys.remove(name)
            self.train_stats_keys.remove(f"train_{name}")
            self.valid_stats_keys.remove(f"val_{name}")

        self._rewrite_csv()

    def visualize(self, epoch, volumes):
        tifdir = os.path.join(
            self.params["dannce_train_dir"], "debug_volumes", f"epoch{epoch}"
        )
        if not os.path.exists(tifdir):
            os.makedirs(tifdir)
        print("Dump training volumes to {}".format(tifdir))
        volumes = volumes.clone().detach().cpu().permute(0, 2, 3, 4, 1).numpy()
        for i in range(volumes.shape[0]):
            for j in range(volumes.shape[-1] // self.params["chan_num"]):
                im = volumes[
                    i,
                    :,
                    :,
                    :,
                    j * self.params["chan_num"] : (j + 1) * self.params["chan_num"],
                ]
                im = norm_im(im) * 255
                im = im.astype("uint8")
                of = os.path.join(tifdir, f"sample{i}" + "_cam" + str(j) + ".tif",)
                imageio.mimwrite(of, np.transpose(im, [2, 0, 1, 3]))
