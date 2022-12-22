import torch
from abc import abstractmethod
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
import os

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, params, model, optimizer, logger, dannce=True):
        # self.config = config
        self.params = params
        self.logger = logger

        self.model = model
        self.optimizer = optimizer

        self.epochs = params['epochs']
        self.save_period = params['save_period']

        self.start_epoch = params.get("start_epoch", 1)

        self.checkpoint_dir = params["dannce_train_dir"] if dannce else params["com_train_dir"]

        # setup visualization writer instance
        logdir = os.path.join(self.checkpoint_dir, "logs")
        if not os.path.exists(logdir):
           os.makedirs(logdir)                
        self.writer = SummaryWriter(log_dir=logdir)

        # self._resume_checkpoint()

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    @abstractmethod
    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            valid_loss, valid_metrics = self._valid_epoch(epoch)

            if epoch % self.save_period == 0 or epoch == self.epochs:
                self._save_checkpoint(epoch)        

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'params': self.params
        }

        # if self.lr_scheduler is not None:
        #     state["lr_scheduler"] = self.lr_scheduler.state_dict()
        if epoch % self.save_period == 0 or epoch == self.epochs:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        else:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint.pth'.format(epoch))
        
        torch.save(state, filename)
        
    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        # if self.params["train_mode"] == "new":
        #     self.logger.info("Train mode set to new. Training from scratch.")
        #     return

        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))