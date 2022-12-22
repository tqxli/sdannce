import torch
import csv, os
from tqdm import tqdm

from dannce.engine.trainer.dannce_trainer import DannceTrainer

class COMTrainer(DannceTrainer):
    def __init__(self, **kwargs):
        super().__init__(dannce=False, **kwargs)

        stats_file = open(os.path.join(self.params["com_train_dir"], "training.csv"), 'w', newline='')
        stats_writer = csv.writer(stats_file)
        self.stats_keys = [*self.loss.names, *self.metrics.names]
        self.train_stats_keys = ["train_"+k for k in self.stats_keys]
        self.valid_stats_keys = ["val_"+k for k in self.stats_keys]
        stats_writer.writerow(["Epoch", *self.train_stats_keys, *self.valid_stats_keys])
        stats_file.close()

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            # open csv
            stats_file = open(os.path.join(self.params["com_train_dir"], "training.csv"), 'a', newline='')
            stats_writer = csv.writer(stats_file)
            stats = [epoch]
            # train
            train_stats = self._train_epoch(epoch)

            for k in self.stats_keys:
                stats.append(train_stats[k])
                    
            result_msg = f"Epoch[{epoch}/{self.epochs}] " \
                + "".join(f"train_{k}: {val:.6f} " for k, val in train_stats.items()) 
            
            # validation
            valid_stats= self._valid_epoch(epoch)

            if len(valid_stats) != 0:
                for k in self.stats_keys:
                    stats.append(valid_stats[k])
                        
                result_msg = result_msg \
                    + "".join(f"val_{k}: {val:.6f} " for k, val in valid_stats.items()) 
            self.logger.info(result_msg)

            # write stats to csv
            stats_writer.writerow(stats)
            stats_file.close()

            # write stats to tensorboard
            for k, v in zip([*self.train_stats_keys, *self.valid_stats_keys], stats[1:]):
                self.writer.add_scalar(k, v, epoch)

            self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        self.model.train()

        # with torch.autograd.set_detect_anomaly(False):
        epoch_loss_dict, epoch_metric_dict = {}, {}
        pbar = tqdm(self.train_dataloader)
        for batch in pbar: 
            self.optimizer.zero_grad()
            
            imgs, gt = batch[0].to(self.device), batch[1].to(self.device)
            pred = self.model(imgs)

            total_loss, loss_dict = self.loss.compute_loss(gt, pred, pred)
            result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {val:.6f} " for loss, val in loss_dict.items())
            pbar.set_description(result)

            total_loss.backward()
            self.optimizer.step()

            epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

            if len(self.metrics.names) != 0: 
                metric_dict = self.metrics.evaluate(pred.detach().cpu().numpy(), gt.clone().cpu().numpy())
                epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)

        if self.lr_scheduler is not None:
            if self.params["lr_scheduler"]["type"] != "ReduceLROnPlateau":
                self.lr_scheduler.step()

        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}

    def _valid_epoch(self, epoch):
        self.model.eval()

        epoch_loss_dict = {}
        epoch_metric_dict = {}

        pbar = tqdm(self.valid_dataloader)
        with torch.no_grad():
            for batch in pbar:
                imgs, gt = batch[0].to(self.device), batch[1].to(self.device)
                pred = self.model(imgs)

                total_loss, loss_dict = self.loss.compute_loss(gt, pred, pred)
                result = f"Epoch[{epoch}/{self.epochs}] " + "".join(f"train_{loss}: {val:.6f} " for loss, val in loss_dict.items())
                pbar.set_description(result)

                epoch_loss_dict = self._update_step(epoch_loss_dict, loss_dict)

                if len(self.metrics.names) != 0: 
                    metric_dict = self.metrics.evaluate(pred.detach().cpu().numpy(), gt.clone().cpu().numpy())
                    epoch_metric_dict = self._update_step(epoch_metric_dict, metric_dict)
        
        if self.lr_scheduler is not None:
            if self.params["lr_scheduler"]["type"] == "ReduceLROnPlateau":
                self.lr_scheduler.step(total_loss)
        
        epoch_loss_dict, epoch_metric_dict = self._average(epoch_loss_dict), self._average(epoch_metric_dict)
        return {**epoch_loss_dict, **epoch_metric_dict}
