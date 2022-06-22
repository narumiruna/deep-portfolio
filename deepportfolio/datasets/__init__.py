import mlconfig
import torch.utils.data

from .dummy import DummyDataset
from .portfolio import PortfolioDataset

mlconfig.register(torch.utils.data.DataLoader)
