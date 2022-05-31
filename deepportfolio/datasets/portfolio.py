from pathlib import Path

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

        self.prices = None
        self.returns = None
        self.prepare_data()

    def prepare_data(self):
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

        prices = pd.concat(series, axis=1)
        prices.dropna(inplace=True)

        returns = prices.pct_change()
        returns.rename(columns={col: col.replace('close', 'return') for col in returns.columns}, inplace=True)

        self.prices = torch.tensor(prices.iloc[1:].values, dtype=torch.float32)
        self.returns = torch.tensor(returns.iloc[1:].values, dtype=torch.float32)

        self.prices = (self.prices - self.prices.mean()) / self.prices.std()

    def __getitem__(self, index):
        x = torch.concat([
            self.prices[index:index + self.window, :],
            self.returns[index:index + self.window, :],
        ],
                         dim=1)

        y = self.returns[index + 1:index + 1 + self.window, :]

        return x, y

    def __len__(self):
        return len(self.prices) - self.window
