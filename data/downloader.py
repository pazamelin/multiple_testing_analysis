#!/usr/bin/python3

import pandas as pd
import numpy as np
import yfinance as yf
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch data from yahoo-finance.')
    parser.add_argument('--tickers', type=str, required=True, help='csv-file with ticker,name columns')
    parser.add_argument('--start',   type=str, required=True, help='date string (YYYY-MM-DD)')
    parser.add_argument('--end',     type=str, required=True, help='date string (YYYY-MM-DD)')
    parser.add_argument('--out',     type=str, required=True, help='the path to the downloaded data')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f'tickers: {args.tickers}')
    print(f'start - {args.start}, end - {args.end}')

    tickers = pd.read_csv(args.tickers, index_col=False)
    info = pd.DataFrame(data=[], columns=['name', 'ticker', 'count'])

    # get data
    tickers_total = len(tickers)
    for index, (ticker, name, _) in tickers.iterrows():
        data = yf.Ticker(ticker)
        data = data.history(start=args.start, end=args.end)
        if len(data) != 0:
            data.to_csv(f'{args.out}/{ticker}.csv')
        # log statistics
        info.loc[index] = [name, ticker, len(data)]
        print(f'[{index}/{tickers_total}] {ticker} - {name}, entries: {len(data)}')

    # dump statistics
    info = info.sort_values(by='count', ascending=False)
    info.to_csv(f'{args.out}/counts.csv')


if __name__ == "__main__":
    if not os.path.exists('downloader-data'):
        os.makedirs('downloader-data')
    main()
