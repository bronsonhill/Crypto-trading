import csv
import pandas as pd


def import_data(file_name, ma=4):
    '''collects specified price data from database and adds a column
    with sma'''
    
    df = pd.read_csv(file_name)
    df.drop(df.columns[[5, 6, 7, 8]], axis=1, inplace=True)

    df.drop('Open', axis=1)
    df.drop('Close', axis=1)

    return df


def averaged_price(df):
    '''mutates df to have column of averaged open, high, low and close'''
    average_price = []

    rows = df.itertuples()
    for row in rows:
        sum_price = float(row[2]) + float(row[3]) + float(row[4]) + float(row[5])
        average_price.append(sum_price/4)
    
    df['Price'] = average_price
    return df


def add_sma(df, length):
    '''from price df adds a column of SMA with specified length'''
    df[f'{length}SMA'] = df['Price'].rolling(length).mean()
    return df


# print(add_sma(import_data('Price_data/EURUSD=X/15m.csv', 4).loc[:,'Close'], 5))

# print(import_data('Price_data/EURUSD=X/15m.csv', 4))

print(add_sma(averaged_price(import_data('Price_data/EURUSD=X/15m.csv')), 4))



def detect_trend(file_name, ma=4):
    '''from ohlc data, returns a list of booleans: True when uptrending
    and False when downtrending'''
    
    trend = []
    df = import_data(file_name, ma)
    
    rows = df.itertuples()
    prev_row = next(rows)
    i = 0
    while i < len(df):
        row = next(rows)
        
        if prev_row > row:
            trend.append(False)
        else:
            trend.append(True)
        
        prev_row = row

    
    return df