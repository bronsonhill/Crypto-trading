import csv
import pandas as pd


def import_data(file_name, ma=0):
    '''collects specified price data from database and adds the '''
    
    df = pd.read_csv(file_name)
    df.drop(df.columns[[5, 6, 7, 8]], axis=1, inplace=True)

    ma_data = [0] * ma
    average_price = []
    trailing_prices = [0] * ma

    rows = df.itertuples()
    i = 0
    for row in rows:
        sum_price = float(row[2]) + float(row[3]) + float(row[4]) + float(row[5])
        average_price.append(sum_price/4)
        
        if i >= ma:
            average = sum(trailing_prices)/ma
            ma_data.append(float(average))

        i += 1
        trailing_prices.insert(0, average_price[-1])
        trailing_prices.pop()

    df.drop('Open', axis=1)
    df.drop('Close', axis=1)

    df[f'{ma}MA'] = ma_data

    return df


def add_sma(price_series, length):
    '''from price series data adds '''
    return price_series.rolling(length).mean()


print(add_sma(import_data('Price_data/EURUSD=X/15m.csv', 4).loc[:,'Close'], 5))

# print(import_data('Price_data/EURUSD=X/15m.csv', 4))





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