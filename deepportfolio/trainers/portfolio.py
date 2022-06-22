from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir

import matplotlib.pyplot as plt
import mlconfig
import mlflow
import pandas as pd
import torch
from loguru import logger
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from tqdm import trange

from ..metrics import Average
from ..metrics import Profit


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
                 test_loader: DataLoader,
                 num_epochs: int = 100):
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
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

        train_set = config.train_set()
        valid_set = config.valid_set()
        test_set = config.test_set()

        train_loader = config.dataloader(dataset=train_set)
        valid_loader = config.dataloader(dataset=valid_set)
        test_loader = config.dataloader(dataset=test_set, sampler=SequentialSampler(test_set), shuffle=False)

        return cls(device=device,
                   model=model,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   train_loader=train_loader,
                   valid_loader=valid_loader,
                   test_loader=test_loader,
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

        self.evaluate()

    def train(self):
        self.model.train()

        loss_meter = Average()
        profit_meter = Profit()

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            out = self.model(x)
            loss = self.loss_fn(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), x.size(0))

            profit_meter.update(out, y)

        self.metrics.update(dict(train_loss=loss_meter.value, train_profit=profit_meter.value))

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        loss_meter = Average()
        profit_meter = Profit()

        for x, y in self.valid_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            out = self.model(x)
            loss = self.loss_fn(out, y)

            loss_meter.update(loss.item(), x.size(0))
            profit_meter.update(out, y)

        self.metrics.update(dict(valid_loss=loss_meter.value, valid_profit=profit_meter.value))

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        daily_returns = []
        daily_weights = []

        for x, y in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            daily_returns.append(y[:, -1, :])
            daily_weights.append(self.model(x)[:, -1, :])

        daily_weights = torch.cat(daily_weights, dim=0)
        daily_returns = torch.cat(daily_returns, dim=0)

        def cum_returns(w):
            if not isinstance(w, torch.Tensor):
                w = torch.as_tensor(w, device=self.device)

            return daily_returns.mul(w).sum(1).add(1).cumprod(dim=0)

        # plot results
        dataset = self.test_loader.dataset
        index = dataset.df.iloc[dataset.window + 1:].index

        # plot resultss
        pd.DataFrame(
            {
                'DLW': cum_returns(daily_weights),
                'Allocation 1: 25/25/25/25': cum_returns([[0.25, 0.25, 0.25, 0.25]]),
                'Allocation_2: 50/10/20/20': cum_returns([[0.5, 0.1, 0.2, 0.2]]),
                'Allocation_3: 10/50/20/20': cum_returns([[0.1, 0.5, 0.2, 0.2]]),
                'Allocation_4: 40/40/10/10': cum_returns([[0.4, 0.40, 0.1, 0.1]]),
                'VTI': cum_returns([[1.0, 0, 0, 0]]),
                'AGG': cum_returns([[0, 1.0, 0, 0]]),
                'DBC': cum_returns([[0, 0, 1.0, 0]]),
                'VIX': cum_returns([[0, 0, 0, 1.0]]),
                'VTI/AGG 80/20': cum_returns([[0.8, 0.2, 0, 0]]),
            },
            index=index).plot(logy=False)

        f = '{}/results.png'.format(gettempdir())
        plt.savefig(f)
        mlflow.log_artifact(f)

        # plot weights
        pd.DataFrame(daily_weights, columns=[col.replace('close', 'weight') for col in dataset.df.columns],
                     index=index).plot(logy=False)
        f = '{}/weights.png'.format(gettempdir())
        plt.savefig(f)
        mlflow.log_artifact(f)

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
