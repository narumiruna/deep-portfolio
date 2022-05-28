import pandas as pd
import yfinance as yf


def main():
    symbols = ['VTI', 'AGG', 'DBC', '^VIX']

    series = []
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='max', interval='1d')

        # close prices
        s = df['Close']
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
