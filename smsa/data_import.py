import numpy as np
import pandas as pd

from typing import Dict


def read_tickers(file) -> pd.DataFrame:
    """
        Returns a dataframe with columns [Ticker, Company] 
    """
    tickers = pd.read_csv(file)
    return tickers


def read_historical_data(tickers, datapath) -> Dict[str, pd.DataFrame]:
    """ 
        Returns a dict {ticker : pd.DataFrame}
        with dataframes containing historical data for a corresponding ticker
    """
    historical_data = {}
    for _, (ticker, *_) in tickers.iterrows():
        # read historical data for a ticker
        ticker_data = pd.read_csv(f'{datapath}/{ticker}.csv')
        # update the resulting dictionary
        historical_data[ticker] = ticker_data
    return historical_data


def merge_table(historical_data, column='Close'):
    """ 
        Returns a table dataframe with columns [Date, Ticker_0, ..., Ticker_N]
        with Ticker_i columns containing @column data for a corresponding ticker
    """
    table_dict = {}

    for ticker, data in historical_data.items():
        # update Date column if required
        if ('Date' not in table_dict) or (len(data['Date']) > len(table_dict['Date'])):
            table_dict['Date'] = data['Date']
        # add @column data
        table_dict[ticker] = data[column]

    return pd.DataFrame.from_dict(table_dict)


def table_logret(table):
    """ 
        For a table dataframe with columns [Date, Ticker_0, ..., Ticker_N]
        convert price data for an every Ticker_i to logarithmic returns column
    """
    for column_lhs in table.columns:
        if column_lhs != 'Date':
            table[column_lhs] = np.log(table[column_lhs]).diff()
    table = table.iloc[1: , :]        
    return table
