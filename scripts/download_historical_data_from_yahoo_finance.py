from pathlib import Path

import click
import yfinance as yf
from tqdm import tqdm


@click.command()
@click.option('-o', '--output-dir', default='data/yfinance')
def main(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    symbols = ['VTI', 'AGG', 'DBC', '^VIX']

    series = []
    for symbol in tqdm(symbols):
        df = yf.Ticker(symbol).history(period='max', interval='1d')

        f = output_dir / '{}.csv'.format(symbol.lstrip('^'))
        df.to_csv(f)


if __name__ == '__main__':
    main()
