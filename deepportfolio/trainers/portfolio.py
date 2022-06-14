from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir

import mlconfig
import mlflow
import torch
from loguru import logger
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange

from ..metrics import Average


@mlconfig.register
class PortfolioTrainer(object):

    def __init__(self,
                 device,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 num_epochs: int = 100):
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs

        self.epoch = 1
        self.metrics = {'best_valid_loss': float('inf')}

    @classmethod
    def from_config(cls, config: mlconfig.Config, **kwargs: dict) -> PortfolioTrainer:
        """Create trainer object from config class

        Args:
            config (mlconfig.Config): trainer config

        Returns PortfolioTrainer object
        """
        device = torch.device(config.device)

        model = config.model()
        model.to(device)
        logger.info('model: {}'.format(model))

        loss_fn = config.loss_fn()
        logger.info('loss function: {}'.format(loss_fn))

        optimizer = config.optimizer(model.parameters())
        logger.info('optimizer: {}'.format(optimizer))

        scheduler = config.scheduler(optimizer)
        logger.info('scheduler: {}'.format(scheduler))

        train_loader = config.train_loader()
        logger.info('train loader: {}'.format(train_loader))

        valid_loader = config.valid_loader()
        logger.info('valid loader: {}'.format(valid_loader))

        return cls(device=device,
                   model=model,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   train_loader=train_loader,
                   valid_loader=valid_loader,
                   **kwargs)

    def fit(self):
        for self.epoch in trange(self.epoch, self.num_epochs + 1):
            self.train()
            self.validate()
            self.scheduler.step()

            self.save_checkpoint()

            mlflow.log_metrics(self.metrics, step=self.epoch)

            format_string = f'Epoch: {self.epoch}/{self.num_epochs}'
            for k, v in self.metrics.items():
                format_string += f', {k}: {v:.4f}'
            logger.info(format_string)

    def train(self):
        self.model.train()

        loss_meter = Average()

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            out = self.model(x)
            loss = self.loss_fn(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), x.size(0))

        self.metrics.update(dict(train_loss=loss_meter.value))

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        loss_meter = Average()

        for x, y in self.valid_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            out = self.model(x)
            loss = self.loss_fn(out, y)

            loss_meter.update(loss.item(), x.size(0))

        self.metrics.update(dict(valid_loss=loss_meter.value))

    def save(self, f: str) -> None:
        """Save checkpoint

        Args:
            f (str): path to save checkpoint
        """
        self.model.eval()

        checkpoint = dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            epoch=self.epoch + 1,
            metrics=self.metrics,
        )

        Path(f).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, f)
        mlflow.log_artifact(f)

    def resume(self, f: str) -> None:
        """Resume from checkpoint

        Args:
            f (str): path to checkpoint
        """
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.epoch = checkpoint['epoch']
        self.metrics = checkpoint['metrics']

    def save_checkpoint(self) -> None:
        """Save best/last checkpoint"""
        if self.metrics['valid_loss'] < self.metrics['best_valid_loss']:
            self.metrics['best_valid_loss'] = self.metrics['valid_loss']
            self.save('{}/best.pth'.format(gettempdir()))
        else:
            self.save('{}/last.pth'.format(gettempdir()))
