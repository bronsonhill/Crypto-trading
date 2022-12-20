from data_processing import sma_data, averaged_price, import_data

import pandas as pd

df = import_data('Price_data/EURUSD=X/15m.csv')
ma_data = sma_data(averaged_price(df), 5)

def find_local_extrema(data, ma=0):
    '''detects local extrema in data, returning a list of tuples with
    index of extrema and extrema value'''
    
    extrema = [(ma-1, data[ma])]
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            # local maximum found
            extrema.append((i, data[i]))
        elif data[i] < data[i-1] and data[i] < data[i+1]:
            # local minimum found
            extrema.append((i, data[i]))
    return extrema


print(find_local_extrema(ma_data, 5))


def price_extrema(df, extrema):
    '''accepts ohlc df and list of extrema and returns a list of tuples
    with index of extrema and price extrema'''

    # if true trend is up, if false trend is down
    trend = None


    return