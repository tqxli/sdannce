import torch
import os, csv
from tqdm import tqdm

from dannce.engine.trainer.dannce_trainer import DANNCETrainer
from dannce.engine.trainer.train_utils import prepare_batch
import dannce.engine.models.loss as custom_losses
from dannce.engine.utils.augmentation import construct_augmented_batch


class SDANNCETrainer(DANNCETrainer):
    def __init__(self, predict_diff=True, relpose=True, **kwargs):
        super().__init__(**kwargs)

        # GCN-specific training options
        self.predict_diff = predict_diff
        self.relpose = relpose

        # adjust loss functions and attributes
        if predict_diff:
            if "WeightedL1Loss" in self.loss.loss_fcns.keys():
                self.loss_sup = self.loss.loss_fcns["WeightedL1Loss"]
            else:
                self.loss_sup = custom_losses.L1Loss()

            self._add_loss_attr(["L1DiffLoss"])
            self._del_loss_attr(["L1Loss"])
            self.loss.loss_fcns.pop("L1Loss")

    def _forward(self, epoch, batch, train=True):
        volumes, grid_centers, keypoints_3d_gt, aux = prepare_batch(batch, self.device)

        # debugging features
        if self.visualize_batch:
            self.visualize(epoch, volumes)
            return

        # form training batch with augmented samples
        if train and self.form_batch:
            volumes, grid_centers, aux = construct_augmented_batch(
                volumes.permute(0, 2, 3, 4, 1),
                grid_centers,
                aux=aux if aux is None else aux.permute(0, 2, 3, 4, 1),
                copies_per_sample=self.form_bs // self.per_batch_sample,
            )
            volumes = volumes.permute(0, 4, 1, 2, 3)
            aux = aux if aux is None else aux.permute(0, 4, 1, 2, 3)

            # update ground truth
            keypoints_3d_gt = (
                keypoints_3d_gt.repeat(self.form_bs // self.per_batch_sample, 1, 1, 1)
                .transpose(1, 0)
                .flatten(0, 1)
            )

        # initial pose generation
        init_poses, keypoints_3d_pred, heatmaps = self.model(volumes, grid_centers)

        if not isinstance(keypoints_3d_pred, list):
            keypoints_3d_gt, keypoints_3d_pred, heatmaps = self._split_data(
                keypoints_3d_gt, keypoints_3d_pred, heatmaps
            )

        return (
            init_poses,
            keypoints_3d_gt,
            keypoints_3d_pred,
            heatmaps,
            grid_centers,
            aux,
        )

    def _forward_loss(
        self,
        init_poses,
        keypoints_3d_gt,
        keypoints_3d_pred,
        heatmaps,
        grid_centers,
        aux,
    ):
        if self.predict_diff and (not self.relpose):
            # estimate absolute offsets
            diff_gt = keypoints_3d_gt - init_poses
            loss_sup = self.loss_sup(diff_gt, keypoints_3d_pred)
            total_loss, loss_dict = self.loss.compute_loss(
                keypoints_3d_gt,
                init_poses + keypoints_3d_pred,
                heatmaps,
                grid_centers,
                aux,
            )
            total_loss += loss_sup
            loss_dict["L1DiffLoss"] = loss_sup.clone().detach().cpu().item()

        elif self.relpose:
            # estimate voxel coordinates
            com3d = torch.mean(grid_centers, dim=1).unsqueeze(-1)  # [N, 3, 1]
            nvox = round(grid_centers.shape[1] ** (1 / 3))
            vsize = (grid_centers[0, :, 0].max() - grid_centers[0, :, 0].min()) / nvox
            keypoints_3d_gt_rel = (keypoints_3d_gt - com3d) / vsize

            if not self.predict_diff:
                loss_sup = self.loss_sup(keypoints_3d_gt_rel, keypoints_3d_pred)

                keypoints_3d_pred = keypoints_3d_pred * vsize + com3d
                total_loss, loss_dict = self.loss.compute_loss(
                    keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux
                )
                total_loss += loss_sup
                loss_dict["L1Loss"] = loss_sup.clone().detach().cpu().item()

            else:
                diff_gt_rel = (keypoints_3d_gt - init_poses) / vsize
                diff_gt_rel_valid = diff_gt_rel[..., : keypoints_3d_pred.shape[-1]]
                diff_loss = self.loss_sup(diff_gt_rel_valid, keypoints_3d_pred)

                # scale back to original, so that bone length loss can be correctly computed
                keypoints_3d_pred = keypoints_3d_pred * vsize  # + com3d
                keypoints_3d_gt = keypoints_3d_gt[..., : keypoints_3d_pred.shape[-1]]
                init_poses = init_poses[..., : keypoints_3d_pred.shape[-1]]

                total_loss, loss_dict = self.loss.compute_loss(
                    keypoints_3d_gt,
                    init_poses + keypoints_3d_pred,
                    heatmaps,
                    grid_centers,
                    aux,
                )
                total_loss += diff_loss
                loss_dict["L1DiffLoss"] = diff_loss.clone().detach().cpu().item()

        else:
            # direct estimation
            total_loss, loss_dict = self.loss.compute_loss(
                keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux
            )

        return total_loss, loss_dict, init_poses, keypoints_3d_gt, keypoints_3d_pred

    def _compute_metrics(self, init_poses, keypoints_3d_gt, keypoints_3d_pred):
        # compute metrics
        if self.predict_diff:
            metric_dict = self.metrics.evaluate(
                (init_poses + keypoints_3d_pred).detach().cpu().numpy(),
                keypoints_3d_gt.clone().cpu().numpy(),
            )
        else:
            metric_dict = self.metrics.evaluate(
                keypoints_3d_pred.detach().cpu().numpy(),
                keypoints_3d_gt.clone().cpu().numpy(),
            )

        return metric_dict

    def _train_epoch(self, epoch):
        self.model.train()

        epoch_loss_dict, epoch_metric_dict = {}, {}
        pbar = tqdm(self.train_dataloader)
        for batch in pbar:
            self.optimizer.zero_grad()

            (
                init_poses,
                keypoints_3d_gt,
                keypoints_3d_pred,
                heatmaps,
                grid_centers,
                aux,
            ) = self._forward(epoch, batch)

            (
                total_loss,
                loss_dict,
                init_poses,
                keypoints_3d_gt,
                keypoints_3d_pred,
            ) = self._forward_loss(
                init_poses,
                keypoints_3d_gt,
                keypoints_3d_pred,
                heatmaps,
                grid_centers,
                aux,
            )

            result = f"Epoch[{epoch}/{self.epochs}] " + "".join(
                f"train_{loss}: {val:.4f} " for loss, val in loss_dict.items()
            )
            pbar.set_description(result)

            total_loss.backward()
            self.optimizer.step()

            epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

            # compute metrics
            if len(self.metrics.names) != 0:
                metric_dict = self._compute_metrics(
                    init_poses, keypoints_3d_gt, keypoints_3d_pred
                )

                epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)

            del total_loss, loss_dict, metric_dict, keypoints_3d_pred, init_poses

        # adjust learning rate
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
                    init_poses,
                    keypoints_3d_gt,
                    keypoints_3d_pred,
                    heatmaps,
                    grid_centers,
                    aux,
                ) = self._forward(epoch, batch, False)

                (
                    total_loss,
                    loss_dict,
                    init_poses,
                    keypoints_3d_gt,
                    keypoints_3d_pred,
                ) = self._forward_loss(
                    init_poses,
                    keypoints_3d_gt,
                    keypoints_3d_pred,
                    heatmaps,
                    grid_centers,
                    aux,
                )

                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                if len(self.metrics.names) != 0:
                    metric_dict = self._compute_metrics(
                        init_poses, keypoints_3d_gt, keypoints_3d_pred
                    )

                    epoch_metric_dict = self._update_step(
                        epoch_metric_dict, metric_dict
                    )

        epoch_loss_dict, epoch_metric_dict = (
            self._average(epoch_loss_dict),
            self._average(epoch_metric_dict),
        )
        return {**epoch_loss_dict, **epoch_metric_dict}
