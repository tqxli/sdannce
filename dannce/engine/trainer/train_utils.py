import dannce.engine.models.loss as custom_losses
import dannce.engine.models.metrics as custom_metrics
import numpy as np
# import pandas as pd

def prepare_batch(batch, device):
    volumes = batch[0].float().to(device)
    grids = batch[1].float().to(device) if batch[1] is not None else None
    targets = batch[2].float().to(device)
    auxs = batch[3].to(device) if batch[3] is not None else None
    
    return volumes, grids, targets, auxs

class LossHelper:
    def __init__(self, params):
        self.loss_params = params
        self._get_losses()

    def _get_losses(self):
        self.loss_fcns = {}
        for name, args in self.loss_params["loss"].items():
            if name == "ConsistencyLoss":
                # params that need to be directly computed from known
                extra_params = {
                    "copies_per_sample": self.loss_params["form_bs"] // self.loss_params["batch_size"],
                }
                self.loss_fcns[name] = getattr(custom_losses, name)(**args, **extra_params)
            else:
                self.loss_fcns[name] = getattr(custom_losses, name)(**args)
        
    def compute_loss(self, kpts_gt, kpts_pred, heatmaps, grid_centers=None, aux=None, heatmaps_gt=None):
        """
        Compute each loss and return their weighted sum for backprop.
        """
        loss_dict = {}
        total_loss = []
        for k, lossfcn in self.loss_fcns.items():
            if k == "GaussianRegLoss":
                loss_val = lossfcn(kpts_gt, kpts_pred.clone().detach(), heatmaps, grid_centers.clone().detach())
            elif k == "MSELoss" or k == "BCELoss":
                # if the 2D heatmaps ground truth are available 
                # and we want to compute the associated 2D losses
                if heatmaps_gt is not None:
                    loss_val = lossfcn(heatmaps_gt, heatmaps)
                else:
                    loss_val = lossfcn(kpts_gt, heatmaps)
            elif 'SilhouetteLoss' in k or k == 'ReconstructionLoss':
                loss_val = lossfcn(aux, heatmaps)
            elif k == 'VarianceLoss':
                loss_val = lossfcn(kpts_pred, heatmaps, grid_centers)
            else:
                loss_val = lossfcn(kpts_gt, kpts_pred)
            total_loss.append(loss_val)
            loss_dict[k] = loss_val.detach().clone().cpu().item()

        return sum(total_loss), loss_dict

    @property
    def names(self):
        return list(self.loss_fcns.keys())

class MetricHelper:
    def __init__(self, params):
        self.metric_names = params["metric"]
        self._get_metrics()

    def _get_metrics(self):
        self.metrics = {}
        for met in self.metric_names:
            self.metrics[met] = getattr(custom_metrics, met)
        
    def evaluate(self, kpts_gt, kpts_pred):
        # perform NaN masking ONCE before metric computation
        metric_dict = {}
        if len(self.metric_names) == 0:
            return metric_dict
            
        kpts_pred, kpts_gt = self.mask_nan(kpts_pred, kpts_gt)
        for met in self.metric_names:
            metric_dict[met] = self.metrics[met](kpts_pred, kpts_gt)

        return metric_dict

    @property
    def names(self):
        return self.metric_names
    
    @classmethod
    def mask_nan(self, pred, gt):
        """
        pred, gt: [bs, 3, n_joints]
        """
        pred = np.transpose(pred.copy(), (1, 0, 2))
        gt = np.transpose(gt.copy(), (1, 0, 2)) #[3, bs, n_joints]
        pred = np.reshape(pred, (pred.shape[0], -1))
        gt = np.reshape(gt, (gt.shape[0], -1)) #[3, bs*n_joints]

        gi = np.where(~np.isnan(np.sum(gt, axis=0)))[0] #[bs*n_joints]

        pred = pred[:, gi]
        gt = gt[:, gi] #[3, bs*n_joints]

        return pred, gt