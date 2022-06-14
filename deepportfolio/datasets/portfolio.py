from pathlib import Path
from typing import Tuple

import mlconfig
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


@mlconfig.register
class PortfolioDataLoader(DataLoader):

    def __init__(self, dataset_params: dict, **kwargs):
        dataset = PortfolioDataset(**dataset_params)
        super(PortfolioDataLoader, self).__init__(dataset, **kwargs)


class PortfolioDataset(Dataset):

    def __init__(self, root: str, window: int = 50, since: str = '2008-01-01', until: str = '2022-01-01'):
        super(Dataset, self).__init__()
        self.root = Path(root)
        self.window = window
        self.since = since
        self.until = until

        self.df = self.read_csv_files()
        self.price_tensor, self.return_tensor = self.prepare_tensors()

    def read_csv_files(self) -> pd.DataFrame:
        csv_files = [csv_file for csv_file in self.root.iterdir() if csv_file.suffix == '.csv']

        series = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, parse_dates=['date'], date_parser=pd.to_datetime)
            df.set_index('date', inplace=True)

            # specify date range
            df = df.loc[self.since:self.until]

            s = df['close']
            s.name = '{}_close'.format(csv_file.stem)
            series.append(s)

        df = pd.concat(series, axis=1)
        df.dropna(inplace=True)

        return df

    def prepare_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        returns = self.df.pct_change()
        returns.rename(columns={col: col.replace('close', 'return') for col in returns.columns}, inplace=True)

        self.price_tensor = torch.tensor(self.df.iloc[1:].values, dtype=torch.float32)
        self.return_tensor = torch.tensor(returns.iloc[1:].values, dtype=torch.float32)

        self.price_tensor = (self.price_tensor - self.price_tensor.mean()) / self.price_tensor.std()
        return self.price_tensor, self.return_tensor

    def __getitem__(self, index):
        x = torch.concat([
            self.price_tensor[index:index + self.window, :],
            self.return_tensor[index:index + self.window, :],
        ],
                         dim=1)

        y = self.return_tensor[index + 1:index + 1 + self.window, :]
        return x, y

    def __len__(self):
        return len(self.price_tensor) - self.window
