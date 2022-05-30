import mlconfig
import mlflow
import torch
from loguru import logger
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange


class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    @property
    def value(self):
        return self.sum / self.count


@mlconfig.register
class PortfolioTrainer(object):

    def __init__(self,
                 device,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: optim.Optimizer,
                 train_loader: DataLoader,
                 num_epochs: int = 100):
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.train_loader = train_loader
        self.num_epochs = num_epochs

        self.epoch = 1

    def fit(self):
        for self.epoch in trange(self.epoch, self.num_epochs + 1):
            self.train()

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

        mlflow.log_metric('train_loss', loss_meter.value, step=self.epoch)
        logger.info('epoch: {}, train_loss: {}'.format(self.epoch, loss_meter.value))
