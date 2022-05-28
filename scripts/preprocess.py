import pandas as pd


def main():
    symbols = ['VTI', 'AGG', 'DBC', 'VIX']

    series = []
    for symbol in symbols:
        csv_file = f'data/{symbol}.csv'

        df = pd.read_csv(csv_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # close prices
        s = df['close']
        s.name = f'{symbol.lower()}_close'
        series.append(s)

        # returns
        s = s.pct_change()
        s.name = f'{symbol.lower()}_return'
        series.append(s)

    df = pd.concat(series, axis=1)
    df.dropna(inplace=True)

    print(df)
    df.to_csv('data/data.csv')


if __name__ == '__main__':
    main()
