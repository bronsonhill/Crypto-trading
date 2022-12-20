from data_processing import add_sma, averaged_price, import_data

import pandas as pd

df = add_sma(averaged_price(import_data('Price_data/EURUSD=X/15m.csv')), 4)

def find_local_extrema(price_series):
    ''''''
    extrema = []
    for i in range(1, len(price_series) - 1):
        if price_series[i] > price_series[i-1] and price_series[i] > price_series[i+1]:
            # local maximum found
            extrema.append((i, price_series[i]))
        elif price_series[i] < price_series[i-1] and price_series[i] < price_series[i+1]:
            # local minimum found
            extrema.append((i, price_series[i]))
    return extrema

print(find_local_extrema(df['4SMA']))

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