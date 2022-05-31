import mlconfig
import mlflow
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
        self.metrics = {}

    def fit(self):
        for self.epoch in trange(self.epoch, self.num_epochs + 1):
            self.train()
            self.validate()
            self.scheduler.step()

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
