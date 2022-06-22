from pathlib import Path
from typing import List
from typing import Tuple

import mlconfig
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..transforms import Normalize


def parse_date(d: str):
    if d is None:
        return None

    return pd.to_datetime(d)


@mlconfig.register
class PortfolioDataset(Dataset):

    def __init__(self,
                 csv_files: List[str],
                 window: int = 50,
                 since: str = None,
                 until: str = None,
                 mean: List[float] = [0.0],
                 std: List[float] = [1.0]):
        super(Dataset, self).__init__()
        self.csv_files = [Path(csv_file) for csv_file in csv_files]
        self.window = window
        self.since = parse_date(since)
        self.until = parse_date(until)
        self.normalize = Normalize(mean=mean, std=std)

        self.df = self.read_csv_files()
        self.price_tensor, self.return_tensor = self.prepare_tensors()

    def read_csv_files(self) -> pd.DataFrame:
        series = []
        for csv_file in self.csv_files:
            df = pd.read_csv(csv_file, parse_dates=['date'], date_parser=pd.to_datetime)
            df.set_index('date', inplace=True)

            s = df['close']
            s.name = '{}_close'.format(csv_file.stem)
            series.append(s)

        df = pd.concat(series, axis=1)
        df.dropna(inplace=True)

        # specify date range
        if self.since is not None:
            df = df.iloc[df.index.get_loc(df.loc[self.since:].index[0]) - self.window:]

        if self.until is not None:
            df = df.loc[:self.until]

        return df

    def prepare_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        returns = self.df.pct_change()
        returns.rename(columns={col: col.replace('close', 'return') for col in returns.columns}, inplace=True)

        self.price_tensor = torch.tensor(self.df.iloc[1:].values, dtype=torch.float32)
        self.return_tensor = torch.tensor(returns.iloc[1:].values, dtype=torch.float32)

        return self.price_tensor, self.return_tensor

    def __getitem__(self, index):
        x = torch.concat([
            self.price_tensor[index:index + self.window, :],
            self.return_tensor[index:index + self.window, :],
        ],
                         dim=1)

        x = self.normalize(x)

        y = self.return_tensor[index + 1:index + 1 + self.window, :]

        assert x.size(0) == y.size(0)

        return x, y

    def __len__(self):
        return len(self.price_tensor) - self.window
