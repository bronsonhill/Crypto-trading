from data_processing import add_sma, averaged_price, import_data

import pandas as pd

df = add_sma(averaged_price(import_data('Price_data/EURUSD=X/15m.csv')), 4)

def find_local_extrema(data):
    '''detects local extrema in data, returning a list of tuples with
    index of extrema and extrema value'''
    
    extrema = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            # local maximum found
            extrema.append((i, data[i]))
        elif data[i] < data[i-1] and data[i] < data[i+1]:
            # local minimum found
            extrema.append((i, data[i]))
    return extrema


print(find_local_extrema(df['4SMA']))